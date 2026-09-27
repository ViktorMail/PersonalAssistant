package com.example.personalassistant.manager

import android.content.Context
import android.util.Log
import java.text.SimpleDateFormat
import java.util.*

class AIServiceManager(private val context: Context) {

    data class AIResponse(val action: String, val response: String, val data: Map<String, String>?)

    private val nlpProcessor = NLPProcessor(context)
    private val networkManager = NetworkManager(context)

    suspend fun processCommand(command: String): AIResponse {
        return try {
            if (networkManager.isOnline()) {
                try {
                    processWithOnlineAI(command)
                } catch (e: Exception) {
                    Log.w("AI_SERVICE", "Online AI failed, using offline: ${e.message}")
                    nlpProcessor.processCommand(command)
                }
            } else {
                nlpProcessor.processCommand(command)
            }
        } catch (e: Exception) {
            AIResponse(
                    action = "error",
                    response = "Извините, не удалось обработать команду. Попробуйте еще раз.",
                    data = null
            )
        }
    }

    private suspend fun processWithOnlineAI(command: String): AIResponse {
        // Здесь будет интеграция с внешним AI API
        // Пока используем оффлайн обработку
        return nlpProcessor.processCommand(command)
    }
}

class NLPProcessor(private val context: Context) {

    fun processCommand(command: String): AIServiceManager.AIResponse {
        val cleanedCommand = command.lowercase(Locale.getDefault())

        return when {
            isAddEventCommand(cleanedCommand) -> parseAddEventCommand(cleanedCommand)
            isShowEventsCommand(cleanedCommand) -> parseShowEventsCommand(cleanedCommand)
            isUpdateEventCommand(cleanedCommand) -> parseUpdateEventCommand(cleanedCommand)
            isDeleteEventCommand(cleanedCommand) -> parseDeleteEventCommand(cleanedCommand)
            isCompleteEventCommand(cleanedCommand) -> parseCompleteEventCommand(cleanedCommand)
            isSyncCommand(cleanedCommand) -> parseSyncCommand(cleanedCommand)
            else -> parseGeneralCommand(cleanedCommand)
        }
    }

    private fun isAddEventCommand(command: String): Boolean {
        val patterns = listOf("добавь", "создай", "запланируй", "новое событие", "новая задача")
        return patterns.any { command.contains(it) }
    }

    private fun parseAddEventCommand(command: String): AIServiceManager.AIResponse {
        val description = extractDescription(command)
        val date = extractDate(command)
        val time = extractTime(command)

        return AIServiceManager.AIResponse(
                action = "add_event",
                response = "Событие \"$description\" добавлено на $date в ${time ?: "течение дня"}",
                data =
                        mapOf(
                                "title" to description,
                                "description" to description,
                                "date" to date,
                                "time" to (time ?: ""),
                                "reminder_time" to calculateReminderTime(time)
                        )
        )
    }

    private fun extractDescription(command: String): String {
        val stopWords =
                listOf(
                        "добавь",
                        "создай",
                        "запланируй",
                        "на",
                        "в",
                        "завтра",
                        "сегодня",
                        "послезавтра",
                        "утром",
                        "днем",
                        "вечером",
                        "встречу",
                        "задачу",
                        "событие"
                )

        var description = command
        stopWords.forEach { word -> description = description.replace(word, "", ignoreCase = true) }

        description = description.replace(Regex("""\d{1,2}[:\.]\d{2}"""), "")
        description = description.replace(Regex("""\d{1,2}[\.\/]\d{1,2}[\.\/]?\d{0,4}"""), "")

        return description.trim().ifEmpty { "Новое событие" }
    }

    private fun extractDate(command: String): String {
        val patterns =
                listOf(
                        Regex("""(\d{1,2})[\.\/](\d{1,2})[\.\/]?(\d{2,4})?"""),
                        Regex("""(завтра)"""),
                        Regex("""(послезавтра)""")
                )

        for (pattern in patterns) {
            val match = pattern.find(command)
            if (match != null) {
                return when (match.value) {
                    "завтра" -> getTomorrowDate()
                    "послезавтра" -> getDayAfterTomorrowDate()
                    else -> parseDateString(match.value)
                }
            }
        }

        return getCurrentDate()
    }

    private fun extractTime(command: String): String? {
        val patterns =
                listOf(
                        Regex("""(\d{1,2})[:\.](\d{2})"""),
                        Regex("""(утром)"""),
                        Regex("""(днем)"""),
                        Regex("""(вечером)""")
                )

        for (pattern in patterns) {
            val match = pattern.find(command)
            if (match != null) {
                return when (match.value) {
                    "утром" -> "09:00"
                    "днем" -> "14:00"
                    "вечером" -> "19:00"
                    else -> parseTimeString(match.value)
                }
            }
        }

        return null
    }

    private fun parseShowEventsCommand(command: String): AIServiceManager.AIResponse {
        return AIServiceManager.AIResponse(
                action = "show_events",
                response = "Показываю список событий",
                data = null
        )
    }

    private fun parseUpdateEventCommand(command: String): AIServiceManager.AIResponse {
        return AIServiceManager.AIResponse(
                action = "update_event",
                response = "Обновляю событие",
                data = null
        )
    }

    private fun parseDeleteEventCommand(command: String): AIServiceManager.AIResponse {
        return AIServiceManager.AIResponse(
                action = "delete_event",
                response = "Удаляю событие",
                data = null
        )
    }

    private fun parseCompleteEventCommand(command: String): AIServiceManager.AIResponse {
        return AIServiceManager.AIResponse(
                action = "complete_event",
                response = "Отмечаю событие как выполненное",
                data = null
        )
    }

    private fun parseSyncCommand(command: String): AIServiceManager.AIResponse {
        return AIServiceManager.AIResponse(
                action = "sync_events",
                response = "Синхронизирую данные с облаком",
                data = null
        )
    }

    private fun parseGeneralCommand(command: String): AIServiceManager.AIResponse {
        return AIServiceManager.AIResponse(
                action = "general",
                response =
                        when {
                            command.contains("привет") -> "Привет! Чем могу помочь?"
                            command.contains("спасибо") -> "Пожалуйста! Обращайтесь еще."
                            command.contains("пока") -> "До свидания! Хорошего дня!"
                            else ->
                                    "Понял вашу команду. Для управления событиями используйте: добавить, показать, удалить."
                        },
                data = null
        )
    }

    private fun parseDateString(dateStr: String): String {
        return try {
            val formats = listOf("dd.MM.yyyy", "dd/MM/yyyy", "dd.MM.yy")
            for (format in formats) {
                try {
                    val date = SimpleDateFormat(format, Locale.getDefault()).parse(dateStr)
                    return SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(date!!)
                } catch (e: Exception) {
                    continue
                }
            }
            getCurrentDate()
        } catch (e: Exception) {
            getCurrentDate()
        }
    }

    private fun parseTimeString(timeStr: String): String {
        return timeStr.replace(".", ":")
    }

    private fun calculateReminderTime(eventTime: String?): String {
        if (eventTime == null) return "09:00"
        return try {
            val timeParts = eventTime.split(":")
            val hour = timeParts[0].toInt()
            val minute = timeParts[1].toInt()

            val reminderMinute = if (minute >= 15) minute - 15 else 45
            val reminderHour = if (minute >= 15) hour else if (hour > 0) hour - 1 else 23

            String.format("%02d:%02d", reminderHour, reminderMinute)
        } catch (e: Exception) {
            "00:00"
        }
    }

    private fun getCurrentDate(): String {
        return SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(Date())
    }

    private fun getTomorrowDate(): String {
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_YEAR, 1)
        return SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(calendar.time)
    }

    private fun getDayAfterTomorrowDate(): String {
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_YEAR, 2)
        return SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(calendar.time)
    }

    private fun isShowEventsCommand(command: String): Boolean = command.contains("покажи")
    private fun isUpdateEventCommand(command: String): Boolean = command.contains("измени")
    private fun isDeleteEventCommand(command: String): Boolean = command.contains("удали")
    private fun isCompleteEventCommand(command: String): Boolean = command.contains("выполн")
    private fun isSyncCommand(command: String): Boolean = command.contains("синхрониз")
}
