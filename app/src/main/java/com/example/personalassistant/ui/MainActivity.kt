package com.example.personalassistant.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.personalassistant.databinding.ActivityMainBinding // ДОБАВЬТЕ ЭТУ СТРОЧКУ
import java.util.Locale

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var binding: ActivityMainBinding
    private lateinit var textToSpeech: TextToSpeech
    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var recognizerIntent: Intent

    private var isListening = false
    private var isTTSInitialized = false

    private val permissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
                    permissions ->
                val allGranted = permissions.values.all { it }
                if (allGranted) {
                    initializeVoiceRecognition()
                    speak("Разрешения предоставлены. Готов к работе!")
                } else {
                    speak("Необходимы разрешения для работы приложения")
                }
            }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initializeTTS()
        setupUI()
        checkPermissions()
    }

    private fun initializeTTS() {
        textToSpeech = TextToSpeech(this, this)
    }

    private fun setupUI() {
        binding.voiceButton.setOnClickListener {
            if (isListening) {
                stopListening()
            } else {
                startListening()
            }
        }

        binding.syncButton.setOnClickListener { syncWithCloud() }

        binding.settingsButton.setOnClickListener { showSettings() }
    }

    private fun checkPermissions() {
        val requiredPermissions =
                mutableListOf(
                        Manifest.permission.RECORD_AUDIO,
                        Manifest.permission.INTERNET,
                        Manifest.permission.ACCESS_NETWORK_STATE
                )

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            requiredPermissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        val missingPermissions =
                requiredPermissions.filter {
                    ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
                }

        if (missingPermissions.isNotEmpty()) {
            permissionLauncher.launch(missingPermissions.toTypedArray())
        } else {
            initializeVoiceRecognition()
        }
    }

    private fun initializeVoiceRecognition() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        recognizerIntent =
                Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(
                            RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                            RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
                    )
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ru-RU")
                    putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                    putExtra(RecognizerIntent.EXTRA_PROMPT, "Скажите команду...")
                }

        speechRecognizer.setRecognitionListener(
                object : RecognitionListener {
                    override fun onReadyForSpeech(params: Bundle) {
                        isListening = true
                        updateListeningState()
                        speak("Слушаю вас")
                    }

                    override fun onBeginningOfSpeech() {}

                    override fun onRmsChanged(rmsdB: Float) {}

                    override fun onBufferReceived(buffer: ByteArray) {}

                    override fun onEndOfSpeech() {
                        isListening = false
                        updateListeningState()
                    }

                    override fun onError(error: Int) {
                        isListening = false
                        updateListeningState()
                        handleRecognitionError(error)
                    }

                    override fun onResults(results: Bundle) {
                        val matches =
                                results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        if (!matches.isNullOrEmpty()) {
                            val command = matches[0]
                            binding.statusText.text = "Распознано: $command"
                            processVoiceCommand(command)
                        }
                    }

                    override fun onPartialResults(partialResults: Bundle) {}
                    override fun onEvent(eventType: Int, params: Bundle) {}
                }
        )
    }

    private fun handleRecognitionError(error: Int) {
        when (error) {
            SpeechRecognizer.ERROR_AUDIO -> speak("Ошибка аудио")
            SpeechRecognizer.ERROR_CLIENT -> speak("Ошибка клиента")
            SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> speak("Недостаточно разрешений")
            else -> speak("Ошибка распознавания речи")
        }
        speak(errorMessage)
        Toast.makeText(this, errorMessage, Toast.LENGTH_SHORT).show()
    }

    private fun processVoiceCommand(command: String) {
        val cleanedCommand = command.lowercase(Locale.getDefault())

        when {
            cleanedCommand.contains("добавь") || cleanedCommand.contains("создай") -> {
                speak("Добавляю новое событие")
                binding.statusText.text = "Добавление события"
                showAddEventDialog()
            }
            cleanedCommand.contains("покажи") || cleanedCommand.contains("показать") -> {
                speak("Показываю список событий")
                binding.statusText.text = "Список событий"
                showEventsList()
            }
            cleanedCommand.contains("удали") || cleanedCommand.contains("удалить") -> {
                speak("Удаляю событие")
                binding.statusText.text = "Удаление события"
            }
            cleanedCommand.contains("привет") -> {
                speak(
                        "Привет! Я ваш голосовой помощник. Скажите 'добавь событие' или 'покажи планы'"
                )
            }
            cleanedCommand.contains("спасибо") -> {
                speak("Пожалуйста! Всегда рад помочь")
            }
            else -> {
                speak("Не понял команду. Попробуйте: добавить, показать или удалить событие")
            }
        }
    }

    private fun showAddEventDialog() {
        // TODO: Implement add event dialog
        Toast.makeText(this, "Функция добавления события", Toast.LENGTH_SHORT).show()
    }

    private fun showEventsList() {
        // TODO: Implement events list display
        Toast.makeText(this, "Функция показа событий", Toast.LENGTH_SHORT).show()
    }

    private fun syncWithCloud() {
        binding.statusText.text = "Синхронизация с облаком..."
        speak("Начинаю синхронизацию с облаком")

        // Имитация синхронизации
        binding.statusText.text = "Синхронизировано с облаком"
        speak("Синхронизация завершена успешно")
    }

    private fun showSettings() {
        startActivity(Intent(this, SetupActivity::class.java))
    }

    private fun startListening() {
        try {
            speechRecognizer.startListening(recognizerIntent)
        } catch (e: Exception) {
            speak("Ошибка запуска распознавания речи")
        }
    }

    private fun stopListening() {
        try {
            speechRecognizer.stopListening()
        } catch (e: Exception) {
            // Игнорируем ошибки при остановке
        }
    }

    private fun updateListeningState() {
        // Use different icons or colors to indicate listening state
        binding.statusText.text = if (isListening) "Слушаю..." else "Готов к работе"
    }

    private fun speak(text: String) {
        if (isTTSInitialized) {
            textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = textToSpeech.setLanguage(Locale("ru", "RU"))
            isTTSInitialized =
                    result != TextToSpeech.LANG_MISSING_DATA &&
                            result != TextToSpeech.LANG_NOT_SUPPORTED

            if (isTTSInitialized) {
                speak("Голосовой помощник активирован. Скажите 'привет' для начала работы")
            } else {
                Toast.makeText(this, "Русский язык не поддерживается", Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        textToSpeech.shutdown()
        speechRecognizer.destroy()
    }
}
