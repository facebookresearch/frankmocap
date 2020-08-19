import os, sys, shutil

def main():
    port_number = sys.argv[1]
    cmd = f"lsof -i :{port_number} > ~/.a.txt"
    os.system(cmd)

    with open('~/.a.txt', "r") as in_f:
        lines = in_f.readlines()
        if len(lines)>0:
            last_line = lines[-1].split()
            pid = int(last_line[1])
            print(f"Killed PID:{pid}")
            cmd = f"kill {pid}"
            os.system(cmd)

    cmd = f"ssh -J snc-fairjmp101 -L {port_number}:localhost:{port_number} devfair283"
    os.system(cmd)

if __name__ == '__main__':
    main()
