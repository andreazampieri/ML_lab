#!/usr/bin/env python3

import re
import sys
import socket
import argparse


def validate_email(email):
    return re.match('[^@]+@(studenti\.)?unitn\.it', email)


def send(email, target_file):
    with open(target_file) as f:
        targets = f.read()

    HOST, PORT = 'lion0b.disi.unitn.it', 7777
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect((HOST, PORT))
        if not validate_email(email):
            print('Error: provide a valid email address')
            sys.exit(1)
        data = '\n'.join([email, targets]) + '\0\n'
        sock.sendall(bytes(data, 'utf-8'))
        anw = str(sock.recv(1024), 'utf-8')
    except (OSError, ConnectionError) as err:
        print('Error: a connection error occurred, '
                'check your internet connection.')
        print(err)
        sys.exit(1)
    finally:
        sock.close()

    if not anw:
        print('Error: a connection error occurred, '
              'check your internet connection.')
        sys.exit(1)

    if anw == 'error':
        print('Error: an unknown error occurred, '
              'please contact paolo.dragone@unitn.it')
        sys.exit(1)
    elif anw == 'format error':
        print('Error: your data is not well formatted')
        sys.exit(1)
    elif anw == 'dataset error':
        print('Error: dataset not recognized')
        sys.exit(1)
    elif anw == 'email error':
        print('Error: provide a valid email address')
        sys.exit(1)
    accuracy, record = anw.splitlines()
    print('Your accuracy = {}'.format(accuracy.strip()))
    print('Current record = {}'.format(record.strip()))


if __name__ == '__main__':
    descr = 'Machine Learning Homework 2 Helper'
    fmt = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=descr, formatter_class=fmt)
    parser.add_argument(
        'email',
        help='Your unitn email'
    )
    parser.add_argument(
        'target_file',
        help='Path to the file containig the predicted test targets'
    )
    args = parser.parse_args()
    send(args.email, args.target_file)

