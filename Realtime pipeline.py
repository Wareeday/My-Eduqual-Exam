"""
Real-Time Signal Processing Pipeline
Implements Apache Kafka streaming and LabStreamingLayer integration
"""

import numpy as np
from kafka import KafkaProducer, KafkaConsumer
import json
import time
from pkg_resources import resource_stream
from pylsl import StreamInfo, StreamOutlet, StreamInlet
from typing import Optional, Callable
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KafkaNeuralStreamer:
    """
    Stream neural data using Apache Kafka
    """
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092', 
                 topic: str = 'neural_signals'):
        """
        Initialize Kafka streamer
        
        Args:
            bootstrap_servers: Kafka server address
            topic: Kafka topic name
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
        self.consumer = None
        
    def create_producer(self):
        """Create Kafka producer"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type='gzip',
                max_request_size=10485760  # 10MB
            )
            logger.info(f"Kafka producer created for topic '{self.topic}'")
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}")
            # Create mock producer for testing without Kafka
            logger.info("Using mock producer for testing")
            self.producer = MockKafkaProducer()
    
    def create_consumer(self, group_id: str = 'bci_consumer_group'):
        """Create Kafka consumer"""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            logger.info(f"Kafka consumer created for topic '{self.topic}'")
        except Exception as e:
            logger.error(f"Failed to create Kafka consumer: {e}")
            # Create mock consumer for testing
            logger.info("Using mock consumer for testing")
            self.consumer = MockKafkaConsumer()
    
    def send_neural_data(self, data: np.ndarray, metadata: dict = None):
        """
        Send neural data to Kafka
        
        Args:
            data: Neural signal data (channels x samples)
            metadata: Additional metadata
        """
        if self.producer is None:
            self.create_producer()
        
        message = {
            'timestamp': time.time(),
            'data': data.tolist(),
            'shape': data.shape,
            'metadata': metadata or {}
        }
        
        self.producer.send(self.topic, value=message)
    
    def consume_neural_data(self, callback: Callable = None, timeout: int = 1000):
        """
        Consume neural data from Kafka
        
        Args:
            callback: Function to process received data
            timeout: Polling timeout in milliseconds
        """
        if self.consumer is None:
            self.create_consumer()
        
        try:
            for message in self.consumer:
                data_dict = message.value
                
                # Reconstruct numpy array
                data = np.array(data_dict['data'])
                
                if callback:
                    callback(data, data_dict.get('metadata', {}))
                else:
                    logger.info(f"Received data with shape {data.shape}")
                    
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
    
    def close(self):
        """Close producer and consumer"""
        if self.producer:
            self.producer.close()
        if self.consumer:
            self.consumer.close()


class MockKafkaProducer:
    """Mock Kafka producer for testing without Kafka server"""
    def send(self, topic, value):
        logger.debug(f"Mock send to {topic}: {value.keys()}")
    
    def close(self):
        pass


class MockKafkaConsumer:
    """Mock Kafka consumer for testing"""
    def __iter__(self):
        return self
    
    def __next__(self):
        time.sleep(0.1)
        # Generate mock data
        mock_message = type('obj', (object,), {
            'value': {
                'timestamp': time.time(),
                'data': np.random.randn(8, 250).tolist(),
                'shape': [8, 250],
                'metadata': {}
            }
        })
        return mock_message
    
    def close(self):
        pass


class LSLNeuralStream:
    """
    Lab Streaming Layer integration for neural data
    """
    
    def __init__(self, stream_name: str = 'BCI_EEG', 
                 stream_type: str = 'EEG',
                 n_channels: int = 8,
                 sampling_rate: float = 250.0):
        """
        Initialize LSL stream
        
        Args:
            stream_name: Name of the LSL stream
            stream_type: Type of stream (EEG, EMG, etc.)
            n_channels: Number of channels
            sampling_rate: Sampling rate in Hz
        """
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.outlet = None
        self.inlet = None
    
    def create_outlet(self, channel_names: list = None):
        """
        Create LSL outlet for sending data
        
        Args:
            channel_names: List of channel names
        """
        info = StreamInfo(
            name=self.stream_name,
            type=self.stream_type,
            channel_count=self.n_channels,
            nominal_srate=self.sampling_rate,
            channel_format='float32',
            source_id='bci_platform_' + str(int(time.time()))
        )
        
        # Add channel information
        if channel_names:
            channels = info.desc().append_child("channels")
            for name in channel_names:
                ch = channels.append_child("channel")
                ch.append_child_value("label", name)
                ch.append_child_value("unit", "microvolts")
                ch.append_child_value("type", "EEG")
        
        self.outlet = StreamOutlet(info)
        logger.info(f"LSL outlet created: {self.stream_name}")
    
    def create_inlet(self, timeout: float = 5.0):
        """
        Create LSL inlet for receiving data
        
        Args:
            timeout: Timeout for finding stream
        """
        logger.info(f"Looking for stream '{self.stream_name}'...")
        streams = resource_stream('name', self.stream_name, timeout=timeout)
        
        if not streams:
            logger.warning(f"No stream found with name '{self.stream_name}'")
            return False
        
        self.inlet = StreamInlet(streams[0])
        logger.info(f"LSL inlet created: {self.stream_name}")
        return True
    
    def push_sample(self, sample: np.ndarray, timestamp: Optional[float] = None):
        """
        Push a sample through LSL outlet
        
        Args:
            sample: Data sample (should match n_channels)
            timestamp: Optional timestamp
        """
        if self.outlet is None:
            raise RuntimeError("Outlet not created. Call create_outlet() first.")
        
        if timestamp:
            self.outlet.push_sample(sample.tolist(), timestamp)
        else:
            self.outlet.push_sample(sample.tolist())
    
    def push_chunk(self, chunk: np.ndarray, timestamps: Optional[list] = None):
        """
        Push a chunk of data through LSL outlet
        
        Args:
            chunk: Data chunk (samples x channels)
            timestamps: Optional list of timestamps
        """
        if self.outlet is None:
            raise RuntimeError("Outlet not created. Call create_outlet() first.")
        
        if timestamps:
            self.outlet.push_chunk(chunk.tolist(), timestamps)
        else:
            self.outlet.push_chunk(chunk.tolist())
    
    def pull_sample(self, timeout: float = 0.0) -> tuple:
        """
        Pull a sample from LSL inlet
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            (sample, timestamp) tuple
        """
        if self.inlet is None:
            raise RuntimeError("Inlet not created. Call create_inlet() first.")
        
        return self.inlet.pull_sample(timeout=timeout)
    
    def pull_chunk(self, timeout: float = 0.0, max_samples: int = 1024) -> tuple:
        """
        Pull a chunk of data from LSL inlet
        
        Args:
            timeout: Timeout in seconds
            max_samples: Maximum samples to retrieve
            
        Returns:
            (samples, timestamps) tuple
        """
        if self.inlet is None:
            raise RuntimeError("Inlet not created. Call create_inlet() first.")
        
        return self.inlet.pull_chunk(timeout=timeout, max_samples=max_samples)


class ParallelProcessor:
    """
    Parallel processing for multi-channel EEG analysis
    """
    
    def __init__(self, n_workers: int = 4):
        """
        Initialize parallel processor
        
        Args:
            n_workers: Number of worker threads
        """
        self.n_workers = n_workers
        self.processing_queue = []
        self.results_queue = []
        
    def process_channel(self, channel_data: np.ndarray, 
                       processing_func: Callable) -> np.ndarray:
        """
        Process a single channel
        
        Args:
            channel_data: Single channel data
            processing_func: Function to apply
            
        Returns:
            Processed data
        """
        return processing_func(channel_data)
    
    def process_parallel(self, data: np.ndarray, 
                        processing_func: Callable) -> np.ndarray:
        """
        Process all channels in parallel
        
        Args:
            data: Multi-channel data (channels x samples)
            processing_func: Function to apply to each channel
            
        Returns:
            Processed multi-channel data
        """
        from concurrent.futures import ThreadPoolExecutor
        
        processed_channels = []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(self.process_channel, data[i], processing_func)
                for i in range(data.shape[0])
            ]
            
            for future in futures:
                processed_channels.append(future.result())
        
        return np.array(processed_channels)


# Example usage and testing
if __name__ == "__main__":
    # Test Kafka streaming
    print("Testing Kafka Neural Streamer...")
    kafka_streamer = KafkaNeuralStreamer()
    
    # Send some data
    test_data = np.random.randn(8, 250)
    kafka_streamer.send_neural_data(test_data, metadata={'trial': 1})
    print("Data sent via Kafka")
    
    # Test LSL streaming
    print("\nTesting LSL Neural Stream...")
    lsl_stream = LSLNeuralStream(n_channels=8, sampling_rate=250.0)
    
    # Create outlet
    channel_names = [f'Ch{i+1}' for i in range(8)]
    lsl_stream.create_outlet(channel_names)
    
    # Push some samples
    for _ in range(5):
        sample = np.random.randn(8)
        lsl_stream.push_sample(sample)
        time.sleep(0.004)  # 250 Hz
    
    print("LSL samples pushed")
    
    # Test parallel processing
    print("\nTesting Parallel Processor...")
    processor = ParallelProcessor(n_workers=4)
    
    def simple_filter(x):
        return x * 0.9  # Simple scaling
    
    data = np.random.randn(8, 1000)
    processed = processor.process_parallel(data, simple_filter)
    print(f"Processed data shape: {processed.shape}")
    
    print("\nAll tests completed!")