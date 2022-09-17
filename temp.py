import threading

class ThreadTest(threading.Thread):
    def __init__(self):
        super(ThreadTest, self).__init__()
        self.counts = 0

    def run(self) -> None:
        for _ in range(100):
            self.counts += 1
            print('counts: ', self.counts)

if __name__ == '__main__':
    file = open('test.txt', 'a')
    file.write('{},{}\n'.format(2, 3))
    file.close()
    # threadTest = ThreadTest()
    # threadTest.start()
