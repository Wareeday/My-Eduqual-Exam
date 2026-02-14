"""
Neural Signal Acquisition Module
Handles EEG signal acquisition from OpenBCI hardware
"""
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
import numpy as np
import serial
import time
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuralSignalAcquisition:
    """
    Handles real-time EEG signal acquisition from OpenBCI
    """
    
    def __init__(self, board_id: int = BoardIds.SYNTHETIC_BOARD, 
                 serial_port: Optional[str] = None):
        """
        Initialize the signal acquisition system
        
        Args:
            board_id: BrainFlow board ID (default: SYNTHETIC for testing)
            serial_port: Serial port for OpenBCI (e.g., 'COM3' or '/dev/ttyUSB0')
        """
        self.board_id = board_id
        self.params = BrainFlowInputParams()
        
        if serial_port:
            self.params.serial_port = serial_port
        
        self.board = BoardShim(board_id, self.params)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.is_streaming = False
        
        logger.info(f"Initialized board {board_id} with sampling rate {self.sampling_rate} Hz")
        logger.info(f"EEG channels: {self.eeg_channels}")
    
    def start_stream(self):
        """Start EEG data streaming"""
        try:
            self.board.prepare_session()
            self.board.start_stream()
            self.is_streaming = True
            logger.info("Started EEG streaming")
        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            raise
    
    def stop_stream(self):
        """Stop EEG data streaming"""
        try:
            if self.is_streaming:
                self.board.stop_stream()
                self.board.release_session()
                self.is_streaming = False
                logger.info("Stopped EEG streaming")
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")
    
    def get_current_data(self, num_samples: int = 250) -> np.ndarray:
        """
        Get current EEG data buffer
        
        Args:
            num_samples: Number of samples to retrieve
            
        Returns:
            EEG data array (channels x samples)
        """
        if not self.is_streaming:
            raise RuntimeError("Stream not started")
        
        data = self.board.get_current_board_data(num_samples)
        eeg_data = data[self.eeg_channels, :]
        return eeg_data
    
    def apply_bandpass_filter(self, data: np.ndarray, 
                             low_freq: float = 0.5, 
                             high_freq: float = 45.0) -> np.ndarray:
        """
        Apply bandpass filter (0.5-45 Hz) to remove artifacts
        
        Args:
            data: EEG data (channels x samples)
            low_freq: Lower cutoff frequency
            high_freq: Upper cutoff frequency
            
        Returns:
            Filtered EEG data
        """
        filtered_data = np.zeros_like(data)
        
        for channel in range(data.shape[0]):
            # Apply bandpass filter
            DataFilter.perform_bandpass(
                data[channel], 
                self.sampling_rate,
                low_freq,
                high_freq,
                order=4,
                filter_type=FilterTypes.BUTTERWORTH.value,
                ripple=0
            )
            filtered_data[channel] = data[channel]
        
        return filtered_data
    
    def apply_notch_filter(self, data: np.ndarray, 
                          notch_freq: float = 50.0) -> np.ndarray:
        """
        Apply notch filter (50/60 Hz) to remove power line noise
        
        Args:
            data: EEG data (channels x samples)
            notch_freq: Notch frequency (50 Hz for EU, 60 Hz for US)
            
        Returns:
            Filtered EEG data
        """
        filtered_data = np.zeros_like(data)
        
        for channel in range(data.shape[0]):
            DataFilter.perform_bandstop(
                data[channel],
                self.sampling_rate,
                notch_freq - 1.0,
                notch_freq + 1.0,
                order=4,
                filter_type=FilterTypes.BUTTERWORTH.value,
                ripple=0
            )
            filtered_data[channel] = data[channel]
        
        return filtered_data
    
    def remove_dc_offset(self, data: np.ndarray) -> np.ndarray:
        """
        Remove DC offset from signals
        
        Args:
            data: EEG data (channels x samples)
            
        Returns:
            DC-corrected data
        """
        return data - np.mean(data, axis=1, keepdims=True)
    
    def preprocess_signal(self, data: np.ndarray, 
                         notch_freq: float = 50.0) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            data: Raw EEG data
            notch_freq: Power line frequency
            
        Returns:
            Preprocessed EEG data
        """
        # Remove DC offset
        data = self.remove_dc_offset(data)
        
        # Apply bandpass filter (0.5-45 Hz)
        data = self.apply_bandpass_filter(data)
        
        # Apply notch filter (50/60 Hz)
        data = self.apply_notch_filter(data, notch_freq)
        
        return data
    
    def get_channel_quality(self, data: np.ndarray) -> dict:
        """
        Assess signal quality for each channel
        
        Args:
            data: EEG data (channels x samples)
            
        Returns:
            Dictionary with quality metrics per channel
        """
        quality = {}
        
        for idx, channel in enumerate(self.eeg_channels):
            channel_data = data[idx]
            
            # Calculate metrics
            quality[f"channel_{channel}"] = {
                "mean": float(np.mean(channel_data)),
                "std": float(np.std(channel_data)),
                "range": float(np.ptp(channel_data)),
                "rms": float(np.sqrt(np.mean(channel_data**2)))
            }
        
        return quality


# Example usage and testing
if __name__ == "__main__":
    # Initialize with synthetic board for testing
    acquisition = NeuralSignalAcquisition(board_id=BoardIds.SYNTHETIC_BOARD)
    
    try:
        # Start streaming
        acquisition.start_stream()
        time.sleep(2)  # Wait for buffer to fill
        
        # Get data
        data = acquisition.get_current_data(num_samples=250)
        print(f"Acquired data shape: {data.shape}")
        
        # Preprocess
        processed_data = acquisition.preprocess_signal(data)
        print(f"Processed data shape: {processed_data.shape}")
        
        # Check quality
        quality = acquisition.get_channel_quality(processed_data)
        print("Signal quality:", quality)
        
    finally:
        acquisition.stop_stream()