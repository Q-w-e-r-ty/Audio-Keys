import gui
import Machine
import os
import sys

def main():
    root = gui.tk.Tk()
    root.configure(bg="#ADD8FF")
    style = gui.ttk.Style(root)
    style.configure('Large.TButton', font=('Helvetica', 24, 'bold'))
    style.configure('Title.TFrame', background="#ADD8FF")
    app = gui.VoiceGuard(root)
    root.mainloop()


if __name__ == '__main__':
    try:
        sys.exit(main())
    finally:
        # This block is crucial to avoid having issues with
        # Python spitting non-sense thread exceptions. We have already
        # handled what we could, so close stderr and stdout.
        if not os.environ.get('CEPH_DEPLOY_TEST'):
            try:
                sys.stdout.close()
            except:
                pass
            try:
                sys.stderr.close()
            except:
                pass