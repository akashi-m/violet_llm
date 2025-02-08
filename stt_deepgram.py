# Copyright 2024 Deepgram SDK contributors. All Rights Reserved.
# Use of this source code is governed by a MIT license that can be found in the LICENSE file.
# SPDX-License-Identifier: MIT

import asyncio
import os
import subprocess

from dotenv import load_dotenv
import logging
from deepgram.utils import verboselogs

from deepgram import (
    DeepgramClient
)

load_dotenv()

SPEAK_TEXT = {"text": "It is - a very pleasant that I can speak with you finaly ) "}
filename = "data/audio_output/test.mp3"


async def main():
    try:
        print("Connecting to Deepgram...")
        # Инициализация клиента
        deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

        response = await deepgram.speak.asyncrest.v("1").save(
            filename, SPEAK_TEXT
        )

        subprocess.run('afplay ' + filename, shell=True)

    except Exception as e:
        print(f"Exception: {e}")

# Ваш текущий код работает нормально
if __name__ == "__main__":
    asyncio.run(main())
