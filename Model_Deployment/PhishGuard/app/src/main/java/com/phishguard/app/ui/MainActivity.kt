package com.phishguard.app.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.*
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import com.phishguard.app.ui.theme.PhishGuardTheme

/**
 * Main entry point for PhishGuard.
 * Single activity with Compose navigation between Scan, History, and Benchmark tabs.
 */
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            PhishGuardTheme {
                PhishGuardApp()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhishGuardApp(viewModel: MainViewModel = viewModel()) {
    var selectedTab by remember { mutableIntStateOf(0) }
    val tabs = listOf(
        TabItem("Scan", Icons.Default.Security),
        TabItem("History", Icons.Default.History),
        TabItem("Benchmark", Icons.Default.Speed),
        TabItem("Evaluate", Icons.Default.Star)
    )

    Scaffold(
        bottomBar = {
            NavigationBar(
                containerColor = MaterialTheme.colorScheme.surface,
                tonalElevation = NavigationBarDefaults.Elevation
            ) {
                tabs.forEachIndexed { index, tab ->
                    NavigationBarItem(
                        icon = { Icon(tab.icon, contentDescription = tab.title) },
                        label = { Text(tab.title) },
                        selected = selectedTab == index,
                        onClick = { selectedTab = index },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = MaterialTheme.colorScheme.primary,
                            selectedTextColor = MaterialTheme.colorScheme.primary,
                            indicatorColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.2f)
                        )
                    )
                }
            }
        }
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            AnimatedContent(
                targetState = selectedTab,
                transitionSpec = {
                    fadeIn() + slideInHorizontally { if (targetState > initialState) it / 4 else -it / 4 } togetherWith
                    fadeOut() + slideOutHorizontally { if (targetState > initialState) -it / 4 else it / 4 }
                },
                label = "tabTransition"
            ) { tab ->
                when (tab) {
                    0 -> ScanScreen(viewModel = viewModel)
                    1 -> HistoryScreen(viewModel = viewModel)
                    2 -> BenchmarkScreen(viewModel = viewModel)
                    3 -> EvaluateScreen(viewModel = viewModel)
                }
            }
        }
    }
}

data class TabItem(
    val title: String,
    val icon: androidx.compose.ui.graphics.vector.ImageVector
)
