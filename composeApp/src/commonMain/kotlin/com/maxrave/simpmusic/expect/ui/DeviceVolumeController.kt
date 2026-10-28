package com.maxrave.simpmusic.expect.ui

import androidx.compose.runtime.Composable

/**
 * Reactive handle to the in-app music playback volume
 * ([com.maxrave.domain.mediaservice.handler.ControlState.volume]), in `0f..1f`.
 *
 * Controls only the app's media player volume (`player.volume` / [com.maxrave.simpmusic.viewModel.UIEvent.UpdateVolume])
 * and never modifies the host device's OS media stream volume (`AudioManager.STREAM_MUSIC`).
 */
interface DeviceVolumeController {
    /** Current in-app music volume normalized to `0f..1f`. */
    val volumeFraction: Float

    /** Sets the in-app music volume from a `0f..1f` fraction without changing device volume. */
    fun setVolumeFraction(fraction: Float)
}

/**
 * Returns a [DeviceVolumeController] bound to the in-app music playback volume.
 */
@Composable
expect fun rememberDeviceVolumeController(): DeviceVolumeController?
