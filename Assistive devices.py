"""
Assistive Device Integration Module
Handles communication with wheelchairs, prosthetics, and communication systems
"""

import numpy as np
import serial
import time
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WheelchairController:
    """
    Controls wheelchair movement based on BCI commands
    """
    
    def __init__(self, serial_port: Optional[str] = None, baudrate: int = 9600):
        """
        Initialize wheelchair controller
        
        Args:
            serial_port: Serial port for Arduino/controller
            baudrate: Serial communication baudrate
        """
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.serial_conn = None
        self.current_state = 'STOP'
        self.speed = 50  # Speed percentage (0-100)
        
        # Command mapping
        self.commands = {
            0: 'STOP',
            1: 'FORWARD',
            2: 'BACKWARD',
            3: 'LEFT',
            4: 'RIGHT'
        }
        
        if serial_port:
            self._connect()
    
    def _connect(self):
        """Establish serial connection"""
        try:
            self.serial_conn = serial.Serial(
                self.serial_port,
                self.baudrate,
                timeout=1
            )
            time.sleep(2)  # Wait for connection to stabilize
            logger.info(f"Connected to wheelchair controller on {self.serial_port}")
        except Exception as e:
            logger.error(f"Failed to connect to wheelchair: {e}")
            self.serial_conn = None
    
    def send_command(self, command_id: int, duration: float = 0.5):
        """
        Send movement command to wheelchair
        
        Args:
            command_id: Command identifier (0-4)
            duration: Duration to execute command (seconds)
        """
        if command_id not in self.commands:
            logger.error(f"Invalid command ID: {command_id}")
            return
        
        command = self.commands[command_id]
        self.current_state = command
        
        # Format: COMMAND,SPEED,DURATION
        message = f"{command},{self.speed},{int(duration*1000)}\n"
        
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(message.encode())
                logger.info(f"Sent command: {message.strip()}")
            except Exception as e:
                logger.error(f"Failed to send command: {e}")
        else:
            # Simulation mode
            logger.info(f"[SIMULATION] Wheelchair command: {message.strip()}")
    
    def set_speed(self, speed: int):
        """
        Set wheelchair speed
        
        Args:
            speed: Speed percentage (0-100)
        """
        self.speed = max(0, min(100, speed))
        logger.info(f"Speed set to {self.speed}%")
    
    def emergency_stop(self):
        """Emergency stop"""
        self.send_command(0, duration=0)
        logger.warning("EMERGENCY STOP activated")
    
    def close(self):
        """Close serial connection"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            logger.info("Wheelchair controller disconnected")


class ProstheticController:
    """
    Controls prosthetic limbs based on neural signals
    """
    
    def __init__(self, device_type: str = 'hand'):
        """
        Initialize prosthetic controller
        
        Args:
            device_type: Type of prosthetic ('hand', 'arm', 'leg')
        """
        self.device_type = device_type
        self.current_pose = 'rest'
        
        # Define poses for different prosthetic types
        self.poses = {
            'hand': ['rest', 'open', 'close', 'pinch', 'point'],
            'arm': ['rest', 'extend', 'flex', 'rotate_cw', 'rotate_ccw'],
            'leg': ['rest', 'step', 'lift', 'extend', 'flex']
        }
        
        logger.info(f"Initialized {device_type} prosthetic controller")
    
    def set_pose(self, pose: str, force: float = 50.0):
        """
        Set prosthetic pose
        
        Args:
            pose: Target pose name
            force: Force percentage (0-100)
        """
        if pose not in self.poses.get(self.device_type, []):
            logger.error(f"Invalid pose '{pose}' for {self.device_type}")
            return
        
        self.current_pose = pose
        
        # In real implementation, this would send PWM signals to motors
        logger.info(f"Prosthetic {self.device_type}: {pose} @ {force}% force")
        
        # Simulate pose execution time
        time.sleep(0.2)
    
    def get_status(self) -> Dict:
        """
        Get current prosthetic status
        
        Returns:
            Status dictionary
        """
        return {
            'device_type': self.device_type,
            'current_pose': self.current_pose,
            'available_poses': self.poses[self.device_type]
        }


class EyeTrackingInterface:
    """
    Eye-tracking integration for multimodal interaction
    """
    
    def __init__(self):
        """Initialize eye-tracking interface"""
        self.gaze_position = (0, 0)
        self.is_calibrated = False
        self.fixation_threshold = 0.5  # seconds
        self.last_fixation_time = 0
        
        logger.info("Eye-tracking interface initialized")
    
    def calibrate(self, calibration_points: list = None):
        """
        Calibrate eye-tracking
        
        Args:
            calibration_points: List of (x, y) calibration points
        """
        if calibration_points is None:
            # Default 9-point calibration
            calibration_points = [
                (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
                (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
                (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
            ]
        
        logger.info(f"Calibrating with {len(calibration_points)} points...")
        
        # Simulate calibration
        for point in calibration_points:
            logger.info(f"  Calibrating point {point}")
            time.sleep(0.5)
        
        self.is_calibrated = True
        logger.info("Calibration complete")
    
    def get_gaze_position(self) -> tuple:
        """
        Get current gaze position
        
        Returns:
            (x, y) normalized gaze coordinates (0-1)
        """
        if not self.is_calibrated:
            logger.warning("Eye-tracking not calibrated")
            return (0.5, 0.5)
        
        # In real implementation, this would read from eye-tracker
        # Simulate gaze movement
        self.gaze_position = (
            np.random.uniform(0, 1),
            np.random.uniform(0, 1)
        )
        
        return self.gaze_position
    
    def detect_fixation(self, current_time: float) -> bool:
        """
        Detect if user is fixating on a point
        
        Args:
            current_time: Current timestamp
            
        Returns:
            True if fixation detected
        """
        # Check if gaze has been stable
        if current_time - self.last_fixation_time > self.fixation_threshold:
            self.last_fixation_time = current_time
            return True
        return False
    
    def get_region_of_interest(self, gaze_pos: tuple, 
                              regions: list) -> Optional[int]:
        """
        Determine which region user is looking at
        
        Args:
            gaze_pos: Current gaze position (x, y)
            regions: List of (x1, y1, x2, y2) region boundaries
            
        Returns:
            Index of region or None
        """
        x, y = gaze_pos
        
        for idx, (x1, y1, x2, y2) in enumerate(regions):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return idx
        
        return None


class CommunicationDevice:
    """
    Alternative and Augmentative Communication (AAC) system
    """
    
    def __init__(self):
        """Initialize communication device"""
        self.vocabulary = [
            "yes", "no", "hello", "thank you", "help",
            "water", "food", "bathroom", "pain", "tired"
        ]
        self.current_message = ""
        self.message_history = []
        
        logger.info("Communication device initialized")
    
    def add_word(self, word: str):
        """
        Add word to current message
        
        Args:
            word: Word to add
        """
        if word in self.vocabulary:
            self.current_message += word + " "
            logger.info(f"Added word: {word}")
        else:
            logger.warning(f"Word '{word}' not in vocabulary")
    
    def speak_message(self, use_tts: bool = True):
        """
        Speak current message
        
        Args:
            use_tts: Use text-to-speech
        """
        if not self.current_message.strip():
            logger.warning("No message to speak")
            return
        
        if use_tts:
            # In real implementation, use pyttsx3 or similar
            logger.info(f"[TTS] Speaking: '{self.current_message.strip()}'")
        else:
            logger.info(f"[TEXT] Message: '{self.current_message.strip()}'")
        
        self.message_history.append(self.current_message.strip())
        self.current_message = ""
    
    def clear_message(self):
        """Clear current message"""
        self.current_message = ""
        logger.info("Message cleared")
    
    def get_vocabulary(self) -> list:
        """Get available vocabulary"""
        return self.vocabulary
    
    def add_to_vocabulary(self, words: list):
        """
        Add words to vocabulary
        
        Args:
            words: List of words to add
        """
        for word in words:
            if word not in self.vocabulary:
                self.vocabulary.append(word)
        logger.info(f"Added {len(words)} words to vocabulary")


class MultimodalInterface:
    """
    Combines BCI, eye-tracking, and other modalities
    """
    
    def __init__(self):
        """Initialize multimodal interface"""
        self.wheelchair = WheelchairController()
        self.prosthetic = ProstheticController()
        self.eye_tracker = EyeTrackingInterface()
        self.comm_device = CommunicationDevice()
        
        self.active_modalities = ['bci', 'eye_tracking']
        
        logger.info("Multimodal interface initialized")
    
    def process_bci_command(self, command_id: int):
        """
        Process BCI command
        
        Args:
            command_id: Command from BCI decoder
        """
        # Route to appropriate device
        if command_id <= 4:
            # Wheelchair control
            self.wheelchair.send_command(command_id)
        else:
            logger.warning(f"Unknown BCI command: {command_id}")
    
    def process_gaze_selection(self, regions: list):
        """
        Process gaze-based selection
        
        Args:
            regions: List of selectable regions
        """
        gaze_pos = self.eye_tracker.get_gaze_position()
        selected_region = self.eye_tracker.get_region_of_interest(gaze_pos, regions)
        
        if selected_region is not None:
            if self.eye_tracker.detect_fixation(time.time()):
                logger.info(f"Selected region: {selected_region}")
                return selected_region
        
        return None
    
    def hybrid_control(self, bci_command: int, gaze_pos: tuple):
        """
        Combine BCI and eye-tracking for refined control
        
        Args:
            bci_command: Command from BCI
            gaze_pos: Gaze position from eye-tracker
        """
        # Example: Use BCI for coarse control, gaze for fine control
        logger.info(f"Hybrid control: BCI={bci_command}, Gaze={gaze_pos}")
        
        # Execute wheelchair command
        self.wheelchair.send_command(bci_command)


# Example usage
if __name__ == "__main__":
    print("Testing Assistive Device Integration...")
    
    # Test wheelchair controller
    print("\n1. Wheelchair Controller:")
    wheelchair = WheelchairController()
    wheelchair.set_speed(75)
    wheelchair.send_command(1, duration=1.0)  # Forward
    time.sleep(1.5)
    wheelchair.send_command(3, duration=0.5)  # Left
    time.sleep(1)
    wheelchair.send_command(0)  # Stop
    
    # Test prosthetic controller
    print("\n2. Prosthetic Controller:")
    prosthetic = ProstheticController(device_type='hand')
    print(f"Status: {prosthetic.get_status()}")
    prosthetic.set_pose('open')
    prosthetic.set_pose('close', force=70)
    prosthetic.set_pose('pinch', force=30)
    
    # Test eye-tracking
    print("\n3. Eye-Tracking Interface:")
    eye_tracker = EyeTrackingInterface()
    eye_tracker.calibrate()
    for _ in range(5):
        gaze = eye_tracker.get_gaze_position()
        print(f"  Gaze position: ({gaze[0]:.2f}, {gaze[1]:.2f})")
        time.sleep(0.5)
    
    # Test communication device
    print("\n4. Communication Device:")
    comm = CommunicationDevice()
    print(f"Vocabulary: {comm.get_vocabulary()[:5]}...")
    comm.add_word("hello")
    comm.add_word("help")
    comm.speak_message()
    
    # Test multimodal interface
    print("\n5. Multimodal Interface:")
    multimodal = MultimodalInterface()
    multimodal.process_bci_command(1)  # Forward
    
    print("\nAll tests completed!")