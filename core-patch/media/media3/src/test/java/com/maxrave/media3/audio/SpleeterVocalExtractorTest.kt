package com.maxrave.media3.audio

import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

class SpleeterVocalExtractorTest {
    @Test
    fun isolatesVocalsAndRejectsDrumsBassAndStereoInstruments() {
        val sampleRate = 44100
        val totalFrames = sampleRate * 2 // 2 seconds
        val extractor = SpleeterVocalExtractor(sampleRate)

        // 1. Pure singing voice (center-panned harmonic series with vibrato: 330 Hz + harmonics up to 2.6 kHz)
        val vocalInL = FloatArray(totalFrames)
        val vocalInR = FloatArray(totalFrames)
        for (n in 0 until totalFrames) {
            val t = n.toDouble() / sampleRate
            val f0 = 330.0 + 4.0 * sin(2.0 * PI * 5.5 * t)
            val v =
                (
                    0.35 * sin(2.0 * PI * f0 * t) +
                        0.30 * sin(2.0 * PI * 2.0 * f0 * t) +
                        0.22 * sin(2.0 * PI * 3.0 * f0 * t) +
                        0.15 * sin(2.0 * PI * 4.0 * f0 * t) +
                        0.10 * sin(2.0 * PI * 5.0 * f0 * t)
                ).toFloat()
            vocalInL[n] = v
            vocalInR[n] = v
        }

        extractor.writeStereo(vocalInL, vocalInR, 0, totalFrames, vocalGain = 1.0f, instrumentGain = 0.0f)
        val vocalAvail = extractor.availableFrames
        assertTrue("Expected separated output frames, got $vocalAvail", vocalAvail > sampleRate)
        val vocalOutL = FloatArray(vocalAvail)
        val vocalOutR = FloatArray(vocalAvail)
        extractor.readStereo(vocalOutL, vocalOutR, 0, vocalAvail)

        val start = vocalAvail / 3
        var vocalEnergySum = 0.0
        for (i in start until vocalAvail) {
            vocalEnergySum += vocalOutL[i] * vocalOutL[i]
        }
        val vocalRms = sqrt(vocalEnergySum / (vocalAvail - start))

        // 2. Full musical accompaniment without vocals:
        //    - Center-panned bassline (65 Hz + 130 Hz)
        //    - Center-panned snare/hi-hat broadband bursts every 0.5s
        //    - Stereo-panned guitar/synth chords (L/R panned)
        extractor.flush()
        val rng = Random(42)
        val accInL = FloatArray(totalFrames)
        val accInR = FloatArray(totalFrames)
        for (n in 0 until totalFrames) {
            val t = n.toDouble() / sampleRate
            val centerBass = (0.35 * sin(2.0 * PI * 65.0 * t) + 0.20 * sin(2.0 * PI * 110.0 * t)).toFloat()
            val guitarL = (0.30 * sin(2.0 * PI * 247.0 * t) + 0.25 * sin(2.0 * PI * 494.0 * t)).toFloat()
            val synthR = (0.30 * sin(2.0 * PI * 370.0 * t) + 0.25 * sin(2.0 * PI * 740.0 * t)).toFloat()
            // Center snare transient for 18ms every 0.5 seconds
            val inSnare = (n % (sampleRate / 2)) < (sampleRate * 18 / 1000)
            val snare = if (inSnare) ((rng.nextDouble() * 2.0 - 1.0) * 0.45).toFloat() else 0f

            accInL[n] = centerBass + snare + 0.85f * guitarL + 0.20f * synthR
            accInR[n] = centerBass + snare + 0.20f * guitarL + 0.85f * synthR
        }

        extractor.writeStereo(accInL, accInR, 0, totalFrames, vocalGain = 1.0f, instrumentGain = 0.0f)
        val accAvail = extractor.availableFrames
        val accOutL = FloatArray(accAvail)
        val accOutR = FloatArray(accAvail)
        extractor.readStereo(accOutL, accOutR, 0, accAvail)

        var accEnergySum = 0.0
        for (i in start until accAvail) {
            accEnergySum += accOutL[i] * accOutL[i]
        }
        val accRms = sqrt(accEnergySum / (accAvail - start))

        assertTrue("Expected vocal RMS ($vocalRms) > 0.15", vocalRms > 0.15)
        assertTrue("Expected accompaniment RMS ($accRms) < 3% of vocal RMS ($vocalRms)", accRms < vocalRms * 0.03)
    }
}
