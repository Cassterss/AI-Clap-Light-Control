#include <Arduino.h>

void setup()
{
    Serial.begin(115200);
    pinMode(5, OUTPUT);
}

void loop()
{
    if (Serial.available())
    {
        String cmd = Serial.readStringUntil('\n');

        if (cmd == "ON")
        {
            digitalWrite(5, HIGH);
            delay(500);
        }

        if (cmd == "OFF")
        {
            digitalWrite(5, LOW);
        }
    }
}

