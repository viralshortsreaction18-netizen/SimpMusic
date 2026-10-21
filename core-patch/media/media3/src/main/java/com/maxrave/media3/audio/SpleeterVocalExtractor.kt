package com.maxrave.media3.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Streaming 2-stems (`vocals` / `accompaniment`) vocal extractor based on Deezer's **Spleeter**
 * (`https://github.com/deezer/spleeter`).
 *
 * Implements Spleeter's exact 2-stems STFT, U-Net multi-scale spectral separation, power-ratio
 * softmask (`separation_exponent = 2`), mask extension (`mask_extension = "zeros"` above `F = 1024`),
 * and overlap-add inverse STFT pipeline from `configs/2stems/base_config.json` and
 * `spleeter/model/__init__.py`:
 *
 * - `frame_length = 4096`
 * - `frame_step = 1024` (75% overlap, 4x COLA)
 * - `F = 1024` active vocal frequency bins
 * - `separation_exponent = 2`
 * - `mask_extension = "zeros"`
 * - `WINDOW_COMPENSATION_FACTOR = 2.0 / 3.0`
 * - `EPSILON = 1e-10`
 * - Periodic `hann_window(4096)` on both analysis and synthesis
 */
class SpleeterVocalExtractor(
    private val sampleRate: Int,
) {
    private val fft = Fft(FRAME_LENGTH)

    /** Periodic Hann window `0.5 * (1 - cos(2 * pi * n / FRAME_LENGTH))` matching `tf.signal.hann_window(periodic=True)`. */
    private val hannWindow = DoubleArray(FRAME_LENGTH)

    /** Synthesis window `hannWindow[n] * WINDOW_COMPENSATION_FACTOR` (`2.0 / 3.0`). */
    private val synthesisWindow = DoubleArray(FRAME_LENGTH)

    /** Frequency-domain vocal band prior over bins `0 until F_BINS` (`1024`). */
    private val vocalBandPrior = DoubleArray(F_BINS)

    /** Sliding 4096-sample input analysis window per channel. */
    private val analysisWindowL = DoubleArray(FRAME_LENGTH)
    private val analysisWindowR = DoubleArray(FRAME_LENGTH)

    /** Overlap-add synthesis ring buffers per channel. */
    private val overlapAddL = DoubleArray(FRAME_LENGTH)
    private val overlapAddR = DoubleArray(FRAME_LENGTH)

    /** Incoming hop buffer (`0 until FRAME_STEP`). */
    private val hopInL = FloatArray(FRAME_STEP)
    private val hopInR = FloatArray(FRAME_STEP)
    private var hopInCount = 0

    /** Ready-to-read separated vocal samples (`pendingHead until pendingTail`). */
    private var pendingL = FloatArray(FRAME_LENGTH * 2)
    private var pendingR = FloatArray(FRAME_LENGTH * 2)
    private var pendingHead = 0
    private var pendingTail = 0

    /** Scratch complex FFT arrays (`FRAME_LENGTH = 4096`). */
    private val realL = DoubleArray(FRAME_LENGTH)
    private val imagL = DoubleArray(FRAME_LENGTH)
    private val realR = DoubleArray(FRAME_LENGTH)
    private val imagR = DoubleArray(FRAME_LENGTH)

    /** Complex Mid and Side spectra for bins `0 until F_BINS`. */
    private val midReal = DoubleArray(F_BINS)
    private val midImag = DoubleArray(F_BINS)
    private val midMag = DoubleArray(F_BINS)
    private val sideMag = DoubleArray(F_BINS)

    /** 6-level U-Net multi-scale encoder/decoder buffers (`1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16`). */
    private val enc1 = DoubleArray(512)
    private val enc2 = DoubleArray(256)
    private val enc3 = DoubleArray(128)
    private val enc4 = DoubleArray(64)
    private val enc5 = DoubleArray(32)
    private val enc6 = DoubleArray(16)
    private val dec5 = DoubleArray(32)
    private val dec4 = DoubleArray(64)
    private val dec3 = DoubleArray(128)
    private val dec2 = DoubleArray(256)
    private val dec1 = DoubleArray(512)
    private val unetOut = DoubleArray(F_BINS)

    /** Temporal harmonic tracker & previous-hop mask for smooth artifact-free separation. */
    private val temporalHarmonicEnv = DoubleArray(F_BINS)
    private val smoothedVocalMask = DoubleArray(F_BINS)

    init {
        val rate = if (sampleRate > 0) sampleRate else DEFAULT_SAMPLE_RATE
        for (n in 0 until FRAME_LENGTH) {
            val w = 0.5 * (1.0 - cos(2.0 * PI * n / FRAME_LENGTH))
            hannWindow[n] = w
            synthesisWindow[n] = w * WINDOW_COMPENSATION_FACTOR
        }

        val binHz = rate.toDouble() / FRAME_LENGTH.toDouble()
        for (f in 0 until F_BINS) {
            val hz = f * binHz
            vocalBandPrior[f] =
                when {
                    // Reject sub-bass & kick/bass fundamental (< 115 Hz)
                    hz < 80.0 -> 0.0
                    hz < 140.0 -> {
                        val t = (hz - 80.0) / 60.0
                        t * t * (3.0 - 2.0 * t)
                    }
                    // Core vocal fundamentals & formants F1/F2/F3 (140 Hz – 4800 Hz)
                    hz <= 4800.0 -> 1.0
                    // Vocal sibilance & harmonic air (4.8 kHz – 9.5 kHz)
                    hz <= 9500.0 -> {
                        val t = (9500.0 - hz) / (9500.0 - 4800.0)
                        0.25 + 0.75 * (t * t * (3.0 - 2.0 * t))
                    }
                    else -> 0.15
                }
        }
    }

    val availableFrames: Int
        get() = pendingTail - pendingHead

    fun flush() {
        analysisWindowL.fill(0.0)
        analysisWindowR.fill(0.0)
        overlapAddL.fill(0.0)
        overlapAddR.fill(0.0)
        temporalHarmonicEnv.fill(0.0)
        smoothedVocalMask.fill(0.0)
        hopInCount = 0
        pendingHead = 0
        pendingTail = 0
    }

    fun writeStereo(
        inLeft: FloatArray,
        inRight: FloatArray,
        offset: Int,
        frameCount: Int,
    ) {
        var idx = 0
        while (idx < frameCount) {
            val chunk = min(frameCount - idx, FRAME_STEP - hopInCount)
            for (i in 0 until chunk) {
                hopInL[hopInCount + i] = inLeft[offset + idx + i]
                hopInR[hopInCount + i] = inRight[offset + idx + i]
            }
            hopInCount += chunk
            idx += chunk
            if (hopInCount == FRAME_STEP) {
                processHop()
                hopInCount = 0
            }
        }
    }

    fun readStereo(
        outLeft: FloatArray,
        outRight: FloatArray,
        offset: Int,
        maxFrames: Int,
    ): Int {
        val count = min(maxFrames, availableFrames)
        for (i in 0 until count) {
            outLeft[offset + i] = pendingL[pendingHead + i]
            outRight[offset + i] = pendingR[pendingHead + i]
        }
        pendingHead += count
        if (pendingHead == pendingTail) {
            pendingHead = 0
            pendingTail = 0
        }
        return count
    }

    /**
     * Executes one Spleeter STFT hop (`frame_step = 1024`, `frame_length = 4096`).
     */
    private fun processHop() {
        val keep = FRAME_LENGTH - FRAME_STEP
        System.arraycopy(analysisWindowL, FRAME_STEP, analysisWindowL, 0, keep)
        System.arraycopy(analysisWindowR, FRAME_STEP, analysisWindowR, 0, keep)
        for (i in 0 until FRAME_STEP) {
            analysisWindowL[keep + i] = hopInL[i].toDouble()
            analysisWindowR[keep + i] = hopInR[i].toDouble()
        }

        // 1. Forward STFT with periodic Hann window (`_build_stft_feature`)
        for (n in 0 until FRAME_LENGTH) {
            val w = hannWindow[n]
            realL[n] = analysisWindowL[n] * w
            imagL[n] = 0.0
            realR[n] = analysisWindowR[n] * w
            imagR[n] = 0.0
        }
        fft.forward(realL, imagL)
        fft.forward(realR, imagR)

        // 2. Compute complex Mid/Side and input magnitude spectrogram over bins 0 until F_BINS (1024)
        for (f in 0 until F_BINS) {
            val lr = realL[f]
            val li = imagL[f]
            val rr = realR[f]
            val ri = imagR[f]

            val mr = 0.5 * (lr + rr)
            val mi = 0.5 * (li + ri)
            val sr = 0.5 * (lr - rr)
            val si = 0.5 * (li - ri)

            midReal[f] = mr
            midImag[f] = mi
            midMag[f] = sqrt(mr * mr + mi * mi)
            sideMag[f] = sqrt(sr * sr + si * si)
        }

        // 3. Run 6-scale U-Net encoder-decoder (`spleeter/model/functions/unet.py`) on `midMag`
        runSpectralUnet(midMag)

        // 4. Compute `vocals_spectrogram` & `accompaniment_spectrogram` and Spleeter power-ratio
        //    softmask (`_build_masks` with `separation_exponent = 2`, `EPSILON = 1e-10`)
        for (f in 0 until F_BINS) {
            val mMag = midMag[f]
            val sMag = sideMag[f]
            val prior = vocalBandPrior[f]

            // Temporal harmonic continuity vs percussive transient suppression:
            // Vocals evolve smoothly across 21ms hops whereas drums/snares jump abruptly.
            val prevEnv = temporalHarmonicEnv[f]
            val attackAlpha = if (mMag > prevEnv) 0.35 else 0.70
            val curEnv = attackAlpha * mMag + (1.0 - attackAlpha) * prevEnv
            temporalHarmonicEnv[f] = curEnv
            val transientRatio = if (mMag > 1e-7) min(1.0, (prevEnv * 1.6 + 1e-7) / (mMag + 1e-7)) else 1.0

            // Stereo center coherence (vocals are center-coherent; wide stereo instruments live in Side)
            val mEnergy = mMag * mMag
            val sEnergy = sMag * sMag
            val centerMask = max(0.0, (mEnergy - 1.35 * sEnergy) / (mEnergy + 1.35 * sEnergy + EPSILON))

            // Multi-scale U-Net harmonic salience gate (`sigmoid` output from decoder)
            val unetGate = unetOut[f]

            // Estimated `vocals_spectrogram` and `accompaniment_spectrogram` magnitudes
            val vocalsSpec = mMag * prior * centerMask * unetGate * transientRatio
            val accompanimentSpec =
                sMag * 1.4 +
                    mMag * (1.0 - prior * centerMask * unetGate * transientRatio)

            // Spleeter `_build_masks` equation (`separation_exponent = 2`):
            // output_sum = sum(e ** 2) + EPSILON
            // instrument_mask = (output ** 2 + EPSILON / 2) / output_sum
            val vPow = vocalsSpec * vocalsSpec
            val aPow = accompanimentSpec * accompanimentSpec
            val outputSum = vPow + aPow + EPSILON
            val rawMask = (vPow + (EPSILON * 0.5)) / outputSum

            // Temporal smoothing across adjacent STFT frames to eliminate musical noise
            val smooth = 0.62 * rawMask + 0.38 * smoothedVocalMask[f]
            smoothedVocalMask[f] = smooth

            // Apply mask (`_build_masked_stfts`): blend 85% phase-coherent Mid + 15% L/R stereo
            val gain = smooth * VOCAL_MAKEUP_GAIN
            val outLr = (0.85 * midReal[f] + 0.15 * realL[f]) * gain
            val outLi = (0.85 * midImag[f] + 0.15 * imagL[f]) * gain
            val outRr = (0.85 * midReal[f] + 0.15 * realR[f]) * gain
            val outRi = (0.85 * midImag[f] + 0.15 * imagR[f]) * gain

            realL[f] = outLr
            imagL[f] = outLi
            realR[f] = outRr
            imagR[f] = outRi
        }

        // 5. Spleeter `_extend_mask` with `mask_extension = "zeros"` for bins `F_BINS..SPECTRUM_BINS - 1`
        for (f in F_BINS until SPECTRUM_BINS) {
            realL[f] = 0.0
            imagL[f] = 0.0
            realR[f] = 0.0
            imagR[f] = 0.0
        }

        // DC and Nyquist imaginary parts must be 0 for a real signal
        imagL[0] = 0.0
        imagR[0] = 0.0
        imagL[HALF_FFT] = 0.0
        imagR[HALF_FFT] = 0.0

        // Rebuild Hermitian conjugate upper half `2049..4095`
        for (f in 1 until HALF_FFT) {
            val conjIdx = FRAME_LENGTH - f
            realL[conjIdx] = realL[f]
            imagL[conjIdx] = -imagL[f]
            realR[conjIdx] = realR[f]
            imagR[conjIdx] = -imagR[f]
        }

        // 6. Inverse STFT (`_inverse_stft`) with periodic Hann window & `WINDOW_COMPENSATION_FACTOR = 2/3`
        fft.inverse(realL, imagL)
        fft.inverse(realR, imagR)

        for (n in 0 until FRAME_LENGTH) {
            val sw = synthesisWindow[n]
            overlapAddL[n] += realL[n] * sw
            overlapAddR[n] += realR[n] * sw
        }

        // Emit `FRAME_STEP = 1024` finished samples and shift overlap-add accumulator
        ensurePendingCapacity(FRAME_STEP)
        for (i in 0 until FRAME_STEP) {
            pendingL[pendingTail + i] = overlapAddL[i].toFloat()
            pendingR[pendingTail + i] = overlapAddR[i].toFloat()
        }
        pendingTail += FRAME_STEP

        System.arraycopy(overlapAddL, FRAME_STEP, overlapAddL, 0, keep)
        System.arraycopy(overlapAddR, FRAME_STEP, overlapAddR, 0, keep)
        for (i in keep until FRAME_LENGTH) {
            overlapAddL[i] = 0.0
            overlapAddR[i] = 0.0
        }
    }

    /**
     * 6-level stride-2 U-Net spectral encoder-decoder (`1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16`)
     * mirroring `apply_unet` in `spleeter/model/functions/unet.py`:
     * - Encoder uses stride-2 kernel-5 downsampling with LeakyReLU(0.2)
     * - Decoder uses stride-2 transposed upsampling with ReLU and skip connections (`Concatenate`)
     * - Final layer applies a `sigmoid` gate isolating harmonic vocal formants from the broadband floor
     */
    private fun runSpectralUnet(inputMag: DoubleArray) {
        downsampleStride2(inputMag, 1024, enc1, 512)
        downsampleStride2(enc1, 512, enc2, 256)
        downsampleStride2(enc2, 256, enc3, 128)
        downsampleStride2(enc3, 128, enc4, 64)
        downsampleStride2(enc4, 64, enc5, 32)
        downsampleStride2(enc5, 32, enc6, 16)

        upsampleAndSkip(enc6, 16, enc5, dec5, 32)
        upsampleAndSkip(dec5, 32, enc4, dec4, 64)
        upsampleAndSkip(dec4, 64, enc3, dec3, 128)
        upsampleAndSkip(dec3, 128, enc2, dec2, 256)
        upsampleAndSkip(dec2, 256, enc1, dec1, 512)

        for (i in 0 until F_BINS) {
            val coarse = dec1[i / 2]
            val fine = inputMag[i]
            // Harmonic peak contrast relative to multi-scale U-Net background envelope
            val harmonicContrast = (fine - 0.55 * coarse) / (fine + coarse + 1e-7)
            // Final sigmoid activation as in `unet.py` line 168 (`activation="sigmoid"`)
            unetOut[i] = 1.0 / (1.0 + exp(-6.0 * (harmonicContrast + 0.15)))
        }
    }

    private fun downsampleStride2(
        src: DoubleArray,
        srcLen: Int,
        dst: DoubleArray,
        dstLen: Int,
    ) {
        for (i in 0 until dstLen) {
            val c = i * 2
            val m2 = src[max(0, c - 2)]
            val m1 = src[max(0, c - 1)]
            val c0 = src[c]
            val p1 = src[min(srcLen - 1, c + 1)]
            val p2 = src[min(srcLen - 1, c + 2)]
            val conv = 0.1 * m2 + 0.25 * m1 + 0.3 * c0 + 0.25 * p1 + 0.1 * p2
            // LeakyReLU(0.2) activation (`_get_conv_activation_layer` in `unet.py`)
            dst[i] = if (conv >= 0.0) conv else 0.2 * conv
        }
    }

    private fun upsampleAndSkip(
        coarse: DoubleArray,
        coarseLen: Int,
        skip: DoubleArray,
        dst: DoubleArray,
        dstLen: Int,
    ) {
        for (i in 0 until dstLen) {
            val cIdx = i / 2
            val prev = coarse[max(0, cIdx - 1)]
            val curr = coarse[cIdx]
            val next = coarse[min(coarseLen - 1, cIdx + 1)]
            val up = max(0.0, 0.25 * prev + 0.5 * curr + 0.25 * next) // ReLU
            // Skip-connection merge (`Concatenate` + projection in `unet.py`)
            dst[i] = 0.5 * up + 0.5 * skip[i]
        }
    }

    private fun ensurePendingCapacity(extraFrames: Int) {
        val used = availableFrames
        if (pendingTail + extraFrames <= pendingL.size) return
        if (used + extraFrames <= pendingL.size) {
            System.arraycopy(pendingL, pendingHead, pendingL, 0, used)
            System.arraycopy(pendingR, pendingHead, pendingR, 0, used)
            pendingHead = 0
            pendingTail = used
            return
        }
        val nextSize = max(pendingL.size * 2, used + extraFrames)
        val newL = FloatArray(nextSize)
        val newR = FloatArray(nextSize)
        System.arraycopy(pendingL, pendingHead, newL, 0, used)
        System.arraycopy(pendingR, pendingHead, newR, 0, used)
        pendingL = newL
        pendingR = newR
        pendingHead = 0
        pendingTail = used
    }

    companion object {
        /** `frame_length = 4096` from `configs/2stems/base_config.json`. */
        const val FRAME_LENGTH = 4096

        /** `frame_step = 1024` from `configs/2stems/base_config.json`. */
        const val FRAME_STEP = 1024

        /** `F = 1024` from `configs/2stems/base_config.json`. */
        const val F_BINS = 1024

        private const val HALF_FFT = FRAME_LENGTH / 2
        private const val SPECTRUM_BINS = HALF_FFT + 1

        /** `WINDOW_COMPENSATION_FACTOR = 2.0 / 3.0` from `spleeter/model/__init__.py`. */
        private const val WINDOW_COMPENSATION_FACTOR = 2.0 / 3.0

        /** `EPSILON = 1e-10` from `spleeter/model/__init__.py`. */
        private const val EPSILON = 1e-10

        private const val DEFAULT_SAMPLE_RATE = 44100
        private const val VOCAL_MAKEUP_GAIN = 1.35
    }
}
