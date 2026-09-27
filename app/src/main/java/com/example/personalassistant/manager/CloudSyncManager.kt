package com.example.personalassistant.manager

import android.content.Context
import android.util.Log
import com.example.personalassistant.data.model.Event
import java.io.*
import java.util.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

class CloudSyncManager(private val context: Context) {

    private val sharedPreferences =
            context.getSharedPreferences("cloud_prefs", Context.MODE_PRIVATE)
    private val networkManager = NetworkManager(context)

    fun initialize() {
        Log.d("CloudSync", "CloudSyncManager initialized")
    }

    fun isCloudSyncEnabled(): Boolean {
        return sharedPreferences.getBoolean("cloud_sync_enabled", false)
    }

    fun getCloudEmail(): String {
        return sharedPreferences.getString("mail_ru_email", "") ?: ""
    }

    suspend fun syncEvents(events: List<Event>): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                if (isCloudSyncEnabled() && networkManager.isOnline()) {
                    val success = uploadEventsToCloud(events)
                    if (success) {
                        Log.d("CloudSync", "Events synced successfully")
                        true
                    } else {
                        false
                    }
                } else {
                    // В оффлайн-режиме сохраняем локально для последующей синхронизации
                    saveEventsForLaterSync(events)
                    true
                }
            } catch (e: Exception) {
                Log.e("CloudSync", "Sync failed: ${e.message}")
                false
            }
        }
    }

    fun testConnection(callback: (Boolean) -> Unit) {
        Thread {
                    try {
                        val email = getCloudEmail()
                        val password = getCloudPassword()

                        if (email.isEmpty() || password.isEmpty()) {
                            callback(false)
                            return@Thread
                        }

                        // Имитация проверки подключения
                        Thread.sleep(1000)
                        val success = true // В реальном приложении здесь будет проверка API

                        callback(success)
                    } catch (e: Exception) {
                        Log.e("CloudSync", "Connection test failed: ${e.message}")
                        callback(false)
                    }
                }
                .start()
    }

    private suspend fun uploadEventsToCloud(events: List<Event>): Boolean {
        return try {
            // Создаем JSON с событиями
            val eventsJson = JSONArray()
            events.forEach { event ->
                val eventJson =
                        JSONObject().apply {
                            put("id", event.id)
                            put("title", event.title)
                            put("description", event.description)
                            put("date", event.date)
                            put("time", event.time ?: "")
                            put("isCompleted", event.isCompleted)
                            put("createdAt", event.createdAt)
                            put("reminderTime", event.reminderTime ?: "")
                            put("category", event.category)
                            put("priority", event.priority)
                        }
                eventsJson.put(eventJson)
            }

            // Сохраняем в локальный файл (эмуляция облачного сохранения)
            val success = saveEventsToLocalFile(eventsJson.toString())

            if (success) {
                // Очищаем очередь отложенной синхронизации
                clearPendingSync()
            }

            success
        } catch (e: Exception) {
            Log.e("CloudSync", "Upload failed: ${e.message}")
            false
        }
    }

    private fun saveEventsToLocalFile(eventsJson: String): Boolean {
        return try {
            val file = File(context.filesDir, "cloud_backup_events.json")
            FileWriter(file).use { writer -> writer.write(eventsJson) }
            true
        } catch (e: Exception) {
            Log.e("CloudSync", "Save to file failed: ${e.message}")
            false
        }
    }

    private fun saveEventsForLaterSync(events: List<Event>) {
        val pendingSync =
                sharedPreferences.getStringSet("pending_sync", mutableSetOf()) ?: mutableSetOf()

        events.forEach { event ->
            val eventData =
                    """
                ${event.id}|${event.title}|${event.description}|${event.date}|${event.time ?: ""}|${event.isCompleted}
            """.trimIndent()
            pendingSync.add(eventData)
        }

        sharedPreferences.edit().putStringSet("pending_sync", pendingSync).apply()
    }

    private fun clearPendingSync() {
        sharedPreferences.edit().remove("pending_sync").apply()
    }

    suspend fun downloadEventsFromCloud(): List<Event> {
        return withContext(Dispatchers.IO) {
            try {
                if (!isCloudSyncEnabled() || !networkManager.isOnline()) {
                    return@withContext emptyList()
                }

                // Загружаем из локального файла (эмуляция облачной загрузки)
                loadEventsFromLocalFile()
            } catch (e: Exception) {
                Log.e("CloudSync", "Download failed: ${e.message}")
                emptyList()
            }
        }
    }

    private fun loadEventsFromLocalFile(): List<Event> {
        return try {
            val file = File(context.filesDir, "cloud_backup_events.json")
            if (!file.exists()) return emptyList()

            val jsonString = FileReader(file).use { it.readText() }
            val eventsJson = JSONArray(jsonString)
            val events = mutableListOf<Event>()

            for (i in 0 until eventsJson.length()) {
                val eventObj = eventsJson.getJSONObject(i)
                events.add(
                        Event(
                                id = eventObj.getLong("id"),
                                title = eventObj.getString("title"),
                                description = eventObj.getString("description"),
                                date = eventObj.getString("date"),
                                time = eventObj.optString("time").takeIf { it.isNotEmpty() },
                                isCompleted = eventObj.getBoolean("isCompleted"),
                                createdAt = eventObj.getLong("createdAt"),
                                reminderTime =
                                        eventObj.optString("reminderTime").takeIf {
                                            it.isNotEmpty()
                                        },
                                category = eventObj.optString("category", "general"),
                                priority = eventObj.optInt("priority", 1)
                        )
                )
            }

            events
        } catch (e: Exception) {
            Log.e("CloudSync", "Load from file failed: ${e.message}")
            emptyList()
        }
    }

    private fun getCloudPassword(): String {
        return sharedPreferences.getString("mail_ru_password", "") ?: ""
    }

    fun getSyncStatus(): String {
        return if (isCloudSyncEnabled()) {
            if (networkManager.isOnline()) "Синхронизировано" else "Ожидание сети"
        } else {
            "Отключено"
        }
    }
}

class NetworkManager(private val context: Context) {

    private val connectivityManager =
            context.getSystemService(Context.CONNECTIVITY_SERVICE) as
                    android.net.ConnectivityManager

    fun isOnline(): Boolean {
        val network = connectivityManager.activeNetwork
        val capabilities = connectivityManager.getNetworkCapabilities(network)
        return capabilities?.hasCapability(
                android.net.NetworkCapabilities.NET_CAPABILITY_INTERNET
        ) == true
    }

    fun getNetworkType(): String {
        val network = connectivityManager.activeNetwork
        val capabilities = connectivityManager.getNetworkCapabilities(network)

        return when {
            capabilities?.hasTransport(android.net.NetworkCapabilities.TRANSPORT_WIFI) == true ->
                    "WiFi"
            capabilities?.hasTransport(android.net.NetworkCapabilities.TRANSPORT_CELLULAR) ==
                    true -> "Мобильная сеть"
            else -> "Нет сети"
        }
    }
}
