package com.example.personalassistant.service

import android.app.*
import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.media.RingtoneManager
import android.os.Build
import androidx.core.app.NotificationCompat
import com.example.personalassistant.R
import com.example.personalassistant.data.model.Event
import com.example.personalassistant.ui.MainActivity
import java.util.*

class NotificationService(private val context: Context) {

    private val notificationManager =
            context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

    companion object {
        const val CHANNEL_ID_EVENTS = "events_channel"
        const val CHANNEL_ID_REMINDERS = "reminders_channel"
        const val NOTIFICATION_ID_EVENT = 1000
    }

    init {
        createNotificationChannels()
    }

    private fun createNotificationChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            // Канал для обычных уведомлений
            val eventsChannel =
                    NotificationChannel(
                                    CHANNEL_ID_EVENTS,
                                    "События",
                                    NotificationManager.IMPORTANCE_DEFAULT
                            )
                            .apply {
                                description = "Уведомления о событиях и действиях"
                                enableLights(true)
                                lightColor = android.graphics.Color.BLUE
                            }

            // Канал для напоминаний
            val remindersChannel =
                    NotificationChannel(
                                    CHANNEL_ID_REMINDERS,
                                    "Напоминания",
                                    NotificationManager.IMPORTANCE_HIGH
                            )
                            .apply {
                                description = "Напоминания о событиях"
                                enableLights(true)
                                enableVibration(true)
                                lightColor = android.graphics.Color.RED
                                vibrationPattern = longArrayOf(0, 500, 250, 500)
                            }

            notificationManager.createNotificationChannel(eventsChannel)
            notificationManager.createNotificationChannel(remindersChannel)
        }
    }

    fun scheduleEventNotification(event: Event) {
        val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val triggerTime = calculateTriggerTime(event)

        if (triggerTime > System.currentTimeMillis()) {
            val intent =
                    Intent(context, NotificationReceiver::class.java).apply {
                        putExtra("event_id", event.id)
                        putExtra("event_title", event.title)
                        putExtra("event_description", event.description)
                        putExtra("event_time", event.time)
                    }

            val pendingIntent =
                    PendingIntent.getBroadcast(
                            context,
                            event.id.toInt(),
                            intent,
                            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
                    )

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                alarmManager.setExactAndAllowWhileIdle(
                        AlarmManager.RTC_WAKEUP,
                        triggerTime,
                        pendingIntent
                )
            } else {
                alarmManager.setExact(AlarmManager.RTC_WAKEUP, triggerTime, pendingIntent)
            }
        }
    }

    private fun calculateTriggerTime(event: Event): Long {
        val calendar =
                Calendar.getInstance().apply {
                    // Парсим дату события
                    val dateParts = event.date.split("-")
                    set(Calendar.YEAR, dateParts[0].toInt())
                    set(Calendar.MONTH, dateParts[1].toInt() - 1)
                    set(Calendar.DAY_OF_MONTH, dateParts[2].toInt())

                    // Устанавливаем время напоминания
                    if (event.reminderTime != null) {
                        val timeParts = event.reminderTime.split(":")
                        set(Calendar.HOUR_OF_DAY, timeParts[0].toInt())
                        set(Calendar.MINUTE, timeParts[1].toInt())
                    } else if (event.time != null) {
                        // Напоминание за 15 минут до события
                        val timeParts = event.time.split(":")
                        set(Calendar.HOUR_OF_DAY, timeParts[0].toInt())
                        set(Calendar.MINUTE, timeParts[1].toInt())
                        add(Calendar.MINUTE, -15)
                    } else {
                        // Напоминание в 9 утра в день события
                        set(Calendar.HOUR_OF_DAY, 9)
                        set(Calendar.MINUTE, 0)
                    }
                    set(Calendar.SECOND, 0)
                }

        return calendar.timeInMillis
    }

    fun showEventReminder(event: Event) {
        val intent =
                Intent(context, MainActivity::class.java).apply {
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
                }

        val pendingIntent =
                PendingIntent.getActivity(
                        context,
                        event.id.toInt(),
                        intent,
                        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
                )

        val notification =
                NotificationCompat.Builder(context, CHANNEL_ID_REMINDERS)
                        .setContentTitle("📅 Напоминание: ${event.title}")
                        .setContentText(event.description)
                        .setSmallIcon(R.drawable.ic_event)
                        .setLargeIcon(
                                BitmapFactory.decodeResource(
                                        context.resources,
                                        R.mipmap.ic_launcher
                                )
                        )
                        .setContentIntent(pendingIntent)
                        .setAutoCancel(true)
                        .setPriority(NotificationCompat.PRIORITY_HIGH)
                        .setDefaults(NotificationCompat.DEFAULT_ALL)
                        .setSound(RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION))
                        .addAction(R.drawable.ic_done, "Выполнено", createCompleteAction(event.id))
                        .addAction(
                                R.drawable.ic_snooze,
                                "Отложить на 1 час",
                                createSnoozeAction(event.id)
                        )
                        .build()

        notificationManager.notify(event.id.toInt() + NOTIFICATION_ID_EVENT, notification)
    }

    private fun createCompleteAction(eventId: Long): PendingIntent {
        val intent =
                Intent(context, NotificationActionReceiver::class.java).apply {
                    action = "COMPLETE_EVENT"
                    putExtra("event_id", eventId)
                }

        return PendingIntent.getBroadcast(
                context,
                eventId.toInt(),
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    private fun createSnoozeAction(eventId: Long): PendingIntent {
        val intent =
                Intent(context, NotificationActionReceiver::class.java).apply {
                    action = "SNOOZE_EVENT"
                    putExtra("event_id", eventId)
                }

        return PendingIntent.getBroadcast(
                context,
                eventId.toInt() + 10000,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    fun cancelNotification(eventId: Long) {
        notificationManager.cancel(eventId.toInt() + NOTIFICATION_ID_EVENT)
    }

    fun showSyncNotification(success: Boolean, message: String? = null) {
        val notification =
                NotificationCompat.Builder(context, CHANNEL_ID_EVENTS)
                        .setContentTitle(
                                if (success) "✅ Синхронизация завершена"
                                else "❌ Ошибка синхронизации"
                        )
                        .setContentText(
                                message
                                        ?: if (success) "Данные успешно синхронизированы"
                                        else "Не удалось синхронизировать данные"
                        )
                        .setSmallIcon(R.drawable.ic_sync)
                        .setAutoCancel(true)
                        .setPriority(NotificationCompat.PRIORITY_DEFAULT)
                        .build()

        notificationManager.notify(999, notification)
    }
}
