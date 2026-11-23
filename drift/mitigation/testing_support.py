"""Shared, test-only helpers used by both the prepared and legacy fixture suites."""

from __future__ import annotations

import socket


def _free_port_block(width: int) -> int:
    for base in range(42000, 60000, width + 3):
        sockets = []
        try:
            for port in range(base, base + width):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sockets.append(sock)
                sock.bind(("127.0.0.1", port))
            return base
        except OSError:
            pass
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("could not reserve a local fixture port block")
