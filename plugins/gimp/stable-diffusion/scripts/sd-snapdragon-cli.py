# !/usr/bin/env python3
"""
Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
"""

import argparse
import os
import random
import shutil
import socket
import subprocess
from pathlib import Path
import logging


class Server:
    def __init__(self, ip: str = "127.0.0.1", timeout: int = 300):
        port = 5001  # fully fixed port

        self.__size_message_length = 16
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(timeout)

        self.s.bind((ip, port))
        self.s.listen(True)

        print(f"[Server]: Listening on {ip}:{port}")
        print("[Server]: Waiting for connection...")
        self.conn, _ = self.s.accept()
        print("[Server]: Connected")

    def __del__(self):
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except Exception as e:
            logging.error(f"[Server]: Error closing socket: {e}")
        try:
            self.s.close()
        except Exception:
            pass

    def close(self):
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except Exception:
            pass
        try:
            self.s.close()
        except Exception:
            pass

    def send(self, message: str):
        message_size = str(len(message)).ljust(self.__size_message_length).encode()
        self.conn.sendall(message_size)
        self.conn.sendall(message.encode())

    def receive(self, decode: bool = True):
        length = self.__receive_value(self.conn, self.__size_message_length)
        if length is not None:
            message = self.__receive_value(self.conn, int(length), decode)
            return message
        return None

    def __receive_value(self, conn, buf_length, decode: bool = True):
        buf = b''
        while buf_length:
            newbuf = conn.recv(buf_length)
            if not newbuf:
                raise ConnectionError("Connection closed by peer")
            buf += newbuf
            buf_length -= len(newbuf)
        return buf.decode() if decode else buf


def initiate_sd_execution(config_path: str, backend: str, workdir: str):
    exe = 'gimp_sd_wos_plugin.exe'
    cmd = [exe, '--config_file_Path', config_path, '--backend', backend]
    print(f"[Launcher] cd {workdir}\n[Launcher] {' '.join(cmd)}")
    return subprocess.Popen(cmd, shell=False, cwd=workdir)


def move_if_exists(src: str, dst: str):
    if os.path.exists(src):
        shutil.move(src, dst)
        return True
    return False


def run_cli(args):
    # Resolve defaults similar to the plugin
    home = str(Path.home())
    workdir = os.path.join(home, 'AppData', 'Roaming', 'GIMP', '2.99', 'plug-ins', 'sd-snapdragon')
    config = 'injected.stablediffusion.properties.in'
    backend = 'QnnHtp.dll'

    os.makedirs(args.outdir, exist_ok=True)

    # Start backend
    backend_process = initiate_sd_execution(config, backend, workdir)

    # Start server and handshake (PORT IS FIXED INSIDE Server)
    server = Server()
    try:
        message = server.receive()
        print("[SERVER RECEIVED]:" + str(message))
        if str(message).lower() != 'initialized':
            raise RuntimeError(f"Unexpected handshake from backend: {message}")

        # Prepare prompt and parameters
        prompt = args.prompt.strip() if args.prompt else ''
        if not prompt:
            raise SystemExit("--prompt is required and cannot be empty")
        if prompt[0] != '#':
            prompt = '#' + prompt
        if len(prompt) <= 1:
            raise SystemExit("--prompt must contain meaningful text, not just whitespace")

        seed = args.seed if args.seed is not None else random.randint(0, 50)
        steps = args.steps
        guidance = args.guidance
        num_images = args.num_images

        # Send configuration to backend
        server.send(f"{guidance}:{seed}:{steps}:{prompt}:{num_images}:")

        produced = []
        for i in range(num_images):
            server.send("execute_next\0")
            msg = server.receive()
            print("[SERVER RECEIVED]:" + str(msg))
            if msg == 'execution_complete':
                # If the backend still writes test.jpeg in workdir, move once without waiting.
                src = os.path.join(workdir, 'test.jpeg')
                out_name = f"sd_{i+1:02d}_seed{seed}_steps{steps}.jpeg"
                dst = os.path.join(args.outdir, out_name)

                if move_if_exists(src, dst):
                    produced.append(dst)
                else:
                    # If you're saving elsewhere now, you can remove this block entirely.
                    raise RuntimeError(f"Expected output {src} not found (no wait).")
            else:
                raise RuntimeError(f"Unexpected message from backend: {msg}")

        server.send("end")

        print(f"[DONE] Generated {len(produced)} image(s).")
        for p in produced:
            print(p)

    finally:
        server.close()
        if backend_process and backend_process.poll() is None:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Run Snapdragon SD backend from terminal')
    p.add_argument('--prompt', required=True, help='Text prompt (will be prefixed with # to match plugin behavior)')
    p.add_argument('--steps', type=int, default=20, help='Number of inference steps (default: 20)')
    p.add_argument('--guidance', type=float, default=7.5, help='Guidance scale (default: 7.5)')
    p.add_argument('--seed', type=int, default=None, help='Seed (default: random 0..50)')
    p.add_argument('--num-images', type=int, default=1, help='Number of images to generate')
    p.add_argument('--outdir', default=str(Path.cwd() / 'outputs'), help='Directory to store generated images')
    return p.parse_args(argv)


if __name__ == '__main__':
    run_cli(parse_args())