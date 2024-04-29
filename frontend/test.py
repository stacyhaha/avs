from multiprocessing import Value
import multiprocessing
import time
#进程间共享变量及列表类型

n=Value('i',0)        #理解成是一个共享对象，i代表整数
def w_f():
    i = 0
    while True:
        n.value= i 
        i += 1
        print("w_f", n.value)
        time.sleep(1)
        

def r_f(n):
    while True:
        print("r", n.value)
        time.sleep(1)


if __name__=='__main__':

        p1=multiprocessing.Process(target=w_f)
        p2=multiprocessing.Process(target=r_f,args=(n,))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
