package com.maxrave.media3.audio

import androidx.media3.common.C
import androidx.media3.common.audio.AudioProcessor
import androidx.media3.common.audio.BaseAudioProcessor
import androidx.media3.common.util.UnstableApi
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

/**
 * Media3 [AudioProcessor] running Deezer's **Spleeter** 2-stems (`vocals` / `accompaniment`)
 * vocal extractor (`https://github.com/deezer/spleeter`).
 *
 * Follows the exact contract established by [EqualizerAudioProcessor] and
 * [ConvolutionReverbAudioProcessor]:
 * - `isActive()` is deliberately NOT overridden so `BaseAudioProcessor` answers it from the format
 *   returned by [onConfigure];
 * - samples [enabled] on every buffer so toggling Spleeter Vocal Extractor mid-song takes effect
 *   immediately on all active and precached ExoPlayer instances;
 * - processes 16-bit PCM in stereo (or mono promoted to dual-mono).
 */
@UnstableApi
class SpleeterVocalAudioProcessor(
    private val enabled: () -> Boolean,
) : BaseAudioProcessor() {
    private var sampleRate = 0
    private var channelCount = 0
    private var extractor: SpleeterVocalExtractor? = null
    private var wasEnabled = false

    private var inLeft = FloatArray(0)
    private var inRight = FloatArray(0)
    private var outLeft = FloatArray(0)
    private var outRight = FloatArray(0)

    override fun onConfigure(inputAudioFormat: AudioProcessor.AudioFormat): AudioProcessor.AudioFormat {
        if (inputAudioFormat.encoding != C.ENCODING_PCM_16BIT) {
            return AudioProcessor.AudioFormat.NOT_SET
        }
        if (inputAudioFormat.channelCount != 1 && inputAudioFormat.channelCount != 2) {
            return AudioProcessor.AudioFormat.NOT_SET
        }
        sampleRate = inputAudioFormat.sampleRate
        channelCount = inputAudioFormat.channelCount
        extractor = SpleeterVocalExtractor(sampleRate)
        wasEnabled = false
        return inputAudioFormat
    }

    override fun onFlush() {
        extractor?.flush()
    }

    override fun onReset() {
        extractor = null
        inLeft = FloatArray(0)
        inRight = FloatArray(0)
        outLeft = FloatArray(0)
        outRight = FloatArray(0)
        sampleRate = 0
        channelCount = 0
        wasEnabled = false
    }

    override fun queueInput(inputBuffer: ByteBuffer) {
        val remaining = inputBuffer.remaining()
        if (remaining == 0) return

        val isNowEnabled = enabled()
        val currentExtractor = extractor
        if (!isNowEnabled || currentExtractor == null) {
            if (wasEnabled) {
                currentExtractor?.flush()
                wasEnabled = false
            }
            val output = replaceOutputBuffer(remaining)
            output.put(inputBuffer)
            output.flip()
            return
        }
        wasEnabled = true

        inputBuffer.order(ByteOrder.nativeOrder())
        val bytesPerFrame = channelCount * SHORT_BYTES
        val frameCount = remaining / bytesPerFrame
        if (frameCount <= 0) {
            inputBuffer.position(inputBuffer.limit())
            return
        }

        ensureScratchCapacity(frameCount)

        if (channelCount == 2) {
            for (i in 0 until frameCount) {
                inLeft[i] = inputBuffer.short * INV_PCM16
                inRight[i] = inputBuffer.short * INV_PCM16
            }
        } else {
            for (i in 0 until frameCount) {
                val sample = inputBuffer.short * INV_PCM16
                inLeft[i] = sample
                inRight[i] = sample
            }
        }

        currentExtractor.writeStereo(inLeft, inRight, 0, frameCount)
        val avail = currentExtractor.availableFrames
        if (avail <= 0) {
            replaceOutputBuffer(0).flip()
            return
        }

        if (outLeft.size < avail) {
            outLeft = FloatArray(avail)
            outRight = FloatArray(avail)
        }
        val readFrames = currentExtractor.readStereo(outLeft, outRight, 0, avail)
        val outputBytes = readFrames * bytesPerFrame
        val output = replaceOutputBuffer(outputBytes)

        if (channelCount == 2) {
            for (i in 0 until readFrames) {
                output.putShort(floatToPcm16(outLeft[i]))
                output.putShort(floatToPcm16(outRight[i]))
            }
        } else {
            for (i in 0 until readFrames) {
                val mono = 0.5f * (outLeft[i] + outRight[i])
                output.putShort(floatToPcm16(mono))
            }
        }
        output.flip()
    }

    private fun ensureScratchCapacity(frames: Int) {
        if (inLeft.size < frames) {
            inLeft = FloatArray(frames)
            inRight = FloatArray(frames)
        }
    }

    private fun floatToPcm16(sample: Float): Short {
        val scaled = (sample * PCM16_SCALE).roundToInt()
        return scaled.coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
    }

    private companion object {
        const val SHORT_BYTES = 2
        const val PCM16_SCALE = 32_768.0f
        const val INV_PCM16 = 1.0f / 32_768.0f
    }
}
