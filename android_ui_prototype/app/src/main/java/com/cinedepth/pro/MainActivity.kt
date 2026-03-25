package com.cinedepth.pro

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.cinedepth.pro.ui.CineDepthApp
import com.cinedepth.pro.ui.theme.CineDepthTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContent {
            CineDepthTheme {
                CineDepthApp()
            }
        }
    }
}
