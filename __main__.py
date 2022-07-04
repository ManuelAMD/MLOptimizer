from app.init_nodes import *
from clear_queues import main
import asyncio
if __name__ == '__main__':
	print("Which node is this instance? (1=master, 2=slave)")
	n = int(input("Type of number"))
	for x in range(4):
		if n == 1:
			InitNodes().master()
		elif n == 2:
			InitNodes().slave()
		else:
			print("Write a valid number")
			loop = asyncio.get_event_loop()
			loop.run_until_complete(main(loop))
			loop.close()

