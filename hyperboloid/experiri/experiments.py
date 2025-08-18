from hyperboloid_aether_os import Contextus
import asyncio
import numpy as np
from simulation.hyperboloid_config import *

async def run_toroid_experiment(mode='standard', turbidities=[1.4], pulses=[0, 128, 255], data='1011010110'*10):
    context = Contextus()
    results = []
    await context.execute_command("CREO 'SIDE_A'")
    await context.execute_command("CREO 'SIDE_B'")
    for turbidity in turbidities:
        TURBIDITY_NTU = turbidity
        for pulse in pulses:
            await context.execute_command(f"SET_LASER SIDE 'A' PULSE {pulse}")
            await context.execute_command(f"SET_LASER SIDE 'B' PULSE {pulse}")
            response = await context.execute_command(f"TOROID '{data}' {'AETHER' if mode == 'aether' else ''}")
            ber_response = await context.execute_command(f"READ_BER {'AETHER' if mode == 'aether' else ''}")
            results.append({
                'mode': mode,
                'turbidity': turbidity,
                'pulse': pulse,
                'response': response,
                'ber': ber_response
            })
            print(f"Experiment ({mode}, Turbidity={turbidity}, Pulse={pulse}): {ber_response}")
    with open('experiment_results.txt', 'a') as f:
        for result in results:
            f.write(f"{time.ctime()}: {result}\n")
    return results

if __name__ == "__main__":
    turbidities = [0.7, 1.4, 2.8]
    pulses = [64, 128, 192]
    asyncio.run(run_toroid_experiment('standard', turbidities, pulses))
    asyncio.run(run_toroid_experiment('aether', turbidities, pulses))
