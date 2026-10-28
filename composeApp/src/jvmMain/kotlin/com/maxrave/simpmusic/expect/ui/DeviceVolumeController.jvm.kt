package com.maxrave.simpmusic.expect.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.maxrave.simpmusic.viewModel.SharedViewModel
import com.maxrave.simpmusic.viewModel.UIEvent
import org.koin.compose.koinInject

@Composable
actual fun rememberDeviceVolumeController(): DeviceVolumeController? {
    val sharedViewModel: SharedViewModel = koinInject()
    val controllerState = sharedViewModel.controllerState.collectAsStateWithLifecycle()

    return remember(sharedViewModel) {
        object : DeviceVolumeController {
            override val volumeFraction: Float
                get() = controllerState.value.volume.coerceIn(0f, 1f)

            override fun setVolumeFraction(fraction: Float) {
                val clamped = fraction.coerceIn(0f, 1f)
                sharedViewModel.onUIEvent(UIEvent.UpdateVolume(clamped))
            }
        }
    }
}
