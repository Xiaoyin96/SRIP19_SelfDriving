import time

print "test log"

print "Start : %s" % time.ctime()
for i in range(10):
    time.sleep(1)
    print i
print "End : %s" % time.ctime()


with open('/home/selfdriving/some_text.txt', 'w') as file:
    file.write('whatever')