package com.example.personalassistant.receiver

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import com.example.personalassistant.PersonalAssistantApp
import com.example.personalassistant.data.model.Event
import com.example.personalassistant.service.NotificationService

class NotificationReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        when {
            intent.action == Intent.ACTION_BOOT_COMPLETED -> {
                // Перепланировать все уведомления после перезагрузки
                rescheduleAllNotifications(context)
            }
            intent.hasExtra("event_id") -> {
                // Показать уведомление о событии
                val eventId = intent.getLongExtra("event_id", 0)
                val title = intent.getStringExtra("event_title") ?: "Событие"
                val description = intent.getStringExtra("event_description") ?: ""
                val time = intent.getStringExtra("event_time")

                val event =
                        Event(
                                id = eventId,
                                title = title,
                                description = description,
                                date = "",
                                time = time,
                                isCompleted = false
                        )

                val notificationService = NotificationService(context)
                notificationService.showEventReminder(event)
            }
        }
    }

    private fun rescheduleAllNotifications(context: Context) {
        // В реальном приложении здесь должна быть логика
        // перепланирования всех активных событий
    }
}

class NotificationActionReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        val eventId = intent.getLongExtra("event_id", 0)
        val app = context.applicationContext as PersonalAssistantApp
        val notificationService = NotificationService(context)

        when (intent.action) {
            "COMPLETE_EVENT" -> {
                // Здесь должна быть логика отметки события как выполненного
                notificationService.cancelNotification(eventId)
                notificationService.showSyncNotification(true, "Событие отмечено как выполненное")
            }
            "SNOOZE_EVENT" -> {
                // Откладываем напоминание на 1 час
                notificationService.cancelNotification(eventId)
                // Здесь должна быть логика перепланирования
                notificationService.showSyncNotification(true, "Напоминание отложено на 1 час")
            }
        }
    }
}
