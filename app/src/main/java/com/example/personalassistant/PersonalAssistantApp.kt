package com.example.personalassistant

import android.app.Application

class PersonalAssistantApp : Application() {

    companion object {
        lateinit var instance: PersonalAssistantApp
            private set
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
    }
}
