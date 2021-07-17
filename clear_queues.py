import asyncio
import aio_pika

#Clean rabbitMQ queues from the broker.
async def main(loop):
	connection = await aio_pika.connect_robust(
		"amqp://{}:{}@{}/".format('guest', 'guest', 'localhost'), loop=loop
	)
	queue_names = ["parameters", "results"]

	async with connection:
		#Create channel
		channel = await connection.channel()

		for queue_name in queue_names:
			queue = await channel.declare_queue(
				queue_name, durable=True
			)
			await queue.purge()

if __name__ == "__main__":
	loop = asyncio.get_event_loop()
	loop.run_until_complete(main(loop))
	loop.close()