package com.maxrave.media3.audio

import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin
import kotlin.math.sqrt

class SpleeterVocalExtractorTest {
    @Test
    fun extractsCenterHarmonicVocalsAndSuppressesSideAccompaniment() {
        val sampleRate = 44100
        val totalFrames = sampleRate * 2 // 2 seconds
        val extractor = SpleeterVocalExtractor(sampleRate)

        val vocalInL = FloatArray(totalFrames)
        val vocalInR = FloatArray(totalFrames)
        for (n in 0 until totalFrames) {
            val t = n.toDouble() / sampleRate
            // Center-panned harmonic vocal (fundamental 440 Hz + 2nd & 3rd harmonics)
            val v = (0.45 * sin(2.0 * PI * 440.0 * t) + 0.25 * sin(2.0 * PI * 880.0 * t) + 0.15 * sin(2.0 * PI * 1320.0 * t)).toFloat()
            vocalInL[n] = v
            vocalInR[n] = v
        }

        extractor.writeStereo(vocalInL, vocalInR, 0, totalFrames)
        val vocalAvail = extractor.availableFrames
        assertTrue("Expected separated output frames, got $vocalAvail", vocalAvail > sampleRate)
        val vocalOutL = FloatArray(vocalAvail)
        val vocalOutR = FloatArray(vocalAvail)
        extractor.readStereo(vocalOutL, vocalOutR, 0, vocalAvail)

        // Measure steady-state RMS in the second half of the output
        val start = vocalAvail / 2
        var vocalEnergySum = 0.0
        for (i in start until vocalAvail) {
            vocalEnergySum += vocalOutL[i] * vocalOutL[i]
        }
        val vocalRms = sqrt(vocalEnergySum / (vocalAvail - start))

        // Now feed pure stereo Side accompaniment (L = -R) + 45 Hz sub-bass (L = R)
        extractor.flush()
        val accInL = FloatArray(totalFrames)
        val accInR = FloatArray(totalFrames)
        for (n in 0 until totalFrames) {
            val t = n.toDouble() / sampleRate
            val side = (0.5 * sin(2.0 * PI * 600.0 * t)).toFloat()
            val subBass = (0.4 * sin(2.0 * PI * 45.0 * t)).toFloat()
            accInL[n] = side + subBass
            accInR[n] = -side + subBass
        }

        extractor.writeStereo(accInL, accInR, 0, totalFrames)
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
        assertTrue("Expected accompaniment RMS ($accRms) << vocal RMS ($vocalRms)", accRms < vocalRms * 0.05)
    }
}
