package com.example.personalassistant.service

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log

class VoiceRecognitionService : Service() {

    private lateinit var speechRecognizer: SpeechRecognizer
    private var isListening = false

    companion object {
        const val ACTION_START_LISTENING = "START_LISTENING"
        const val ACTION_STOP_LISTENING = "STOP_LISTENING"
        const val EXTRA_COMMAND_RESULT = "COMMAND_RESULT"
    }

    override fun onCreate() {
        super.onCreate()
        initializeSpeechRecognizer()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_LISTENING -> startListening()
            ACTION_STOP_LISTENING -> stopListening()
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun initializeSpeechRecognizer() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)

        speechRecognizer.setRecognitionListener(
                object : RecognitionListener {
                    override fun onReadyForSpeech(params: android.os.Bundle?) {
                        isListening = true
                        Log.d("VoiceService", "Ready for speech")
                    }

                    override fun onBeginningOfSpeech() {
                        Log.d("VoiceService", "Beginning of speech")
                    }

                    override fun onRmsChanged(rmsdB: Float) {
                        // Можно использовать для визуализации громкости
                    }

                    override fun onBufferReceived(buffer: ByteArray?) {}

                    override fun onEndOfSpeech() {
                        isListening = false
                        Log.d("VoiceService", "End of speech")
                    }

                    override fun onError(error: Int) {
                        isListening = false
                        Log.e("VoiceService", "Recognition error: $error")
                        broadcastError(error)
                    }

                    override fun onResults(results: android.os.Bundle?) {
                        val matches =
                                results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        if (!matches.isNullOrEmpty()) {
                            val command = matches[0]
                            Log.d("VoiceService", "Recognized: $command")
                            broadcastCommandResult(command)
                        }
                    }

                    override fun onPartialResults(partialResults: android.os.Bundle?) {}
                    override fun onEvent(eventType: Int, params: android.os.Bundle?) {}
                }
        )
    }

    private fun startListening() {
        if (!isListening) {
            val intent =
                    Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                        putExtra(
                                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
                        )
                        putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ru-RU")
                        putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
                        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                    }
            speechRecognizer.startListening(intent)
        }
    }

    private fun stopListening() {
        if (isListening) {
            speechRecognizer.stopListening()
        }
    }

    private fun broadcastCommandResult(command: String) {
        val intent =
                Intent(ACTION_START_LISTENING).apply { putExtra(EXTRA_COMMAND_RESULT, command) }
        sendBroadcast(intent)
    }

    private fun broadcastError(errorCode: Int) {
        val intent = Intent(ACTION_STOP_LISTENING).apply { putExtra("error_code", errorCode) }
        sendBroadcast(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        speechRecognizer.destroy()
    }
}
