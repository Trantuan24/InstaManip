import pynvml
import logging
import torch
import traceback

logger = logging.getLogger(__name__)


class GPUMonitor:
    def __init__(self, disable: bool = False):
        self.disable_nvml = disable
        self.disable_msg = (
            "NVML stats are turned off manually and will be reported as -1 values."
        )

        if not self.disable_nvml:
            pynvml.nvmlInit()
            self.device = torch.cuda.current_device()
            assert self.device >= 0
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)

    def _collect_nvml_stats(self):
    # -> Tuple[float, float, float, float]:
        utilization, memory_used, temperature, power = -1, -1, -1, -1

        if self.disable_nvml:
            logger.warn(self.disable_msg)
            return utilization, memory_used, temperature, power

        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            memory_used = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1024**3
            temperature = pynvml.nvmlDeviceGetTemperature(
                self.handle, sensor=pynvml.NVML_TEMPERATURE_GPU
            )
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000
        except Exception as e:
            self.disable_nvml = True
            logger.error(
                "NVML stats collection failed with the exception and will not be retried later. "
                "Stats are filled with -1 from now on. Exception below."
            )
            logger.error(traceback.format_exc())
            self.disable_msg = "NVML stats collection is disabled due to previous failure and will not be retried."
        return utilization, memory_used, temperature, power

    def get_stats(self):
    # -> Dict[str, float]:
        """
        https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html#structnvmlUtilization__t
        https://docs.nvidia.com/deploy/nvml-api/structnvmlMemory__t.html#structnvmlMemory__t
        """
        utilization, memory_used, temperature, power = self._collect_nvml_stats()

        mem_stats = torch.cuda.memory_stats()
        return {
            "gpu_usage": utilization,
            "used_memory_gb": memory_used,
            "power": power,
            "temperature": temperature,
            "active_gb": mem_stats["active_bytes.all.peak"] / 1024**3,
            "allocated_gb": mem_stats["allocated_bytes.all.peak"] / 1024**3,
            "reserved_gb": mem_stats["reserved_bytes.all.peak"] / 1024**3,
            "num_alloc_retries": mem_stats["num_alloc_retries"],
        }
    