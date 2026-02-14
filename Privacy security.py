"""
Privacy and Security Module for Neural Data
Implements encryption, anonymization, and consent management
"""

import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC 
from cryptography.hazmat.backends import default_backend
import hashlib
import json
import base64
import time
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuralDataEncryption:
    """
    Handles encryption and decryption of neural data
    """
    
    def __init__(self, password: Optional[str] = None):
        """
        Initialize encryption system
        
        Args:
            password: Password for key derivation (if None, generates key)
        """
        if password:
            self.key = self._derive_key_from_password(password)
        else:
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
        logger.info("Encryption system initialized")
    
    def _derive_key_from_password(self, password: str, 
                                  salt: bytes = b'neural_bci_salt') -> bytes:
        """
        Derive encryption key from password
        
        Args:
            password: User password
            salt: Salt for key derivation
            
        Returns:
            Derived key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_data(self, data: np.ndarray) -> bytes:
        """
        Encrypt neural data
        
        Args:
            data: Neural signal data
            
        Returns:
            Encrypted data as bytes
        """
        # Convert numpy array to bytes
        data_bytes = data.tobytes()
        
        # Add metadata about shape and dtype
        metadata = {
            'shape': data.shape,
            'dtype': str(data.dtype)
        }
        metadata_str = json.dumps(metadata)
        
        # Combine metadata and data
        combined = metadata_str.encode() + b'|||' + data_bytes
        
        # Encrypt
        encrypted = self.cipher.encrypt(combined)
        
        return encrypted
    
    def decrypt_data(self, encrypted_data: bytes) -> np.ndarray:
        """
        Decrypt neural data
        
        Args:
            encrypted_data: Encrypted bytes
            
        Returns:
            Decrypted numpy array
        """
        # Decrypt
        decrypted = self.cipher.decrypt(encrypted_data)
        
        # Split metadata and data
        parts = decrypted.split(b'|||', 1)
        metadata_str = parts[0].decode()
        data_bytes = parts[1]
        
        # Parse metadata
        metadata = json.loads(metadata_str)
        
        # Reconstruct array
        data = np.frombuffer(data_bytes, dtype=metadata['dtype'])
        data = data.reshape(metadata['shape'])
        
        return data
    
    def get_key(self) -> bytes:
        """Get encryption key"""
        return self.key
    
    def save_key(self, filepath: str):
        """
        Save encryption key to file
        
        Args:
            filepath: Path to save key
        """
        with open(filepath, 'wb') as f:
            f.write(self.key)
        logger.info(f"Encryption key saved to {filepath}")
    
    def load_key(self, filepath: str):
        """
        Load encryption key from file
        
        Args:
            filepath: Path to key file
        """
        with open(filepath, 'rb') as f:
            self.key = f.read()
        self.cipher = Fernet(self.key)
        logger.info(f"Encryption key loaded from {filepath}")


class DataAnonymizer:
    """
    Anonymizes neural data and metadata
    """
    
    def __init__(self):
        """Initialize anonymizer"""
        self.anonymization_map = {}
        logger.info("Data anonymizer initialized")
    
    def generate_pseudonym(self, user_id: str) -> str:
        """
        Generate pseudonym for user ID
        
        Args:
            user_id: Original user identifier
            
        Returns:
            Pseudonymized ID
        """
        if user_id not in self.anonymization_map:
            # Generate deterministic but irreversible pseudonym
            hash_obj = hashlib.sha256(user_id.encode())
            pseudonym = hash_obj.hexdigest()[:16]
            self.anonymization_map[user_id] = pseudonym
        
        return self.anonymization_map[user_id]
    
    def anonymize_metadata(self, metadata: Dict) -> Dict:
        """
        Remove or pseudonymize identifying information
        
        Args:
            metadata: Original metadata
            
        Returns:
            Anonymized metadata
        """
        anonymized = metadata.copy()
        
        # Fields to remove
        sensitive_fields = ['name', 'email', 'phone', 'address', 'ssn']
        
        for field in sensitive_fields:
            if field in anonymized:
                del anonymized[field]
        
        # Pseudonymize user ID if present
        if 'user_id' in anonymized:
            anonymized['user_id'] = self.generate_pseudonym(anonymized['user_id'])
        
        # Generalize age
        if 'age' in anonymized:
            age = anonymized['age']
            age_group = f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
            anonymized['age_group'] = age_group
            del anonymized['age']
        
        # Remove precise timestamps, keep only date
        if 'timestamp' in anonymized:
            timestamp = anonymized['timestamp']
            date_only = time.strftime('%Y-%m-%d', time.localtime(timestamp))
            anonymized['date'] = date_only
            del anonymized['timestamp']
        
        return anonymized
    
    def apply_differential_privacy(self, data: np.ndarray, 
                                   epsilon: float = 1.0) -> np.ndarray:
        """
        Add noise for differential privacy
        
        Args:
            data: Original data
            epsilon: Privacy parameter (smaller = more private)
            
        Returns:
            Data with added noise
        """
        # Calculate noise scale based on sensitivity and epsilon
        sensitivity = np.max(np.abs(data))
        noise_scale = sensitivity / epsilon
        
        # Add Laplacian noise
        noise = np.random.laplace(0, noise_scale, data.shape)
        noisy_data = data + noise
        
        logger.info(f"Applied differential privacy with epsilon={epsilon}")
        
        return noisy_data


class ConsentManager:
    """
    Manages user consent for data collection and usage
    """
    
    def __init__(self):
        """Initialize consent manager"""
        self.consent_records = {}
        logger.info("Consent manager initialized")
    
    def record_consent(self, user_id: str, consent_type: str, 
                      granted: bool, purposes: list = None):
        """
        Record user consent
        
        Args:
            user_id: User identifier
            consent_type: Type of consent (data_collection, research, sharing)
            granted: Whether consent was granted
            purposes: List of specific purposes
        """
        if user_id not in self.consent_records:
            self.consent_records[user_id] = {}
        
        self.consent_records[user_id][consent_type] = {
            'granted': granted,
            'timestamp': time.time(),
            'purposes': purposes or [],
            'version': '1.0'
        }
        
        logger.info(f"Consent recorded for user {user_id}: "
                   f"{consent_type} = {granted}")
    
    def check_consent(self, user_id: str, consent_type: str) -> bool:
        """
        Check if user has granted consent
        
        Args:
            user_id: User identifier
            consent_type: Type of consent to check
            
        Returns:
            True if consent granted
        """
        if user_id not in self.consent_records:
            return False
        
        if consent_type not in self.consent_records[user_id]:
            return False
        
        return self.consent_records[user_id][consent_type]['granted']
    
    def revoke_consent(self, user_id: str, consent_type: str):
        """
        Revoke user consent
        
        Args:
            user_id: User identifier
            consent_type: Type of consent to revoke
        """
        if user_id in self.consent_records:
            if consent_type in self.consent_records[user_id]:
                self.consent_records[user_id][consent_type]['granted'] = False
                self.consent_records[user_id][consent_type]['revoked_at'] = time.time()
                
                logger.info(f"Consent revoked for user {user_id}: {consent_type}")
    
    def get_consent_summary(self, user_id: str) -> Dict:
        """
        Get consent summary for user
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with consent status
        """
        if user_id not in self.consent_records:
            return {}
        
        return self.consent_records[user_id]
    
    def export_consent_record(self, user_id: str) -> str:
        """
        Export consent record as JSON
        
        Args:
            user_id: User identifier
            
        Returns:
            JSON string of consent record
        """
        record = self.get_consent_summary(user_id)
        return json.dumps(record, indent=2)


class SecureTransmission:
    """
    Handles secure transmission of neural data
    """
    
    def __init__(self):
        """Initialize secure transmission"""
        self.encryption = NeuralDataEncryption()
        logger.info("Secure transmission initialized")
    
    def prepare_for_transmission(self, data: np.ndarray, 
                                 metadata: Dict) -> Dict:
        """
        Prepare data package for transmission
        
        Args:
            data: Neural data
            metadata: Associated metadata
            
        Returns:
            Secure data package
        """
        # Encrypt data
        encrypted_data = self.encryption.encrypt_data(data)
        
        # Create checksum for integrity
        checksum = hashlib.sha256(encrypted_data).hexdigest()
        
        # Create package
        package = {
            'encrypted_data': base64.b64encode(encrypted_data).decode(),
            'metadata': metadata,
            'checksum': checksum,
            'timestamp': time.time(),
            'version': '1.0'
        }
        
        return package
    
    def receive_transmission(self, package: Dict) -> tuple:
        """
        Receive and verify transmitted data
        
        Args:
            package: Received data package
            
        Returns:
            (data, metadata) tuple
        """
        # Decode encrypted data
        encrypted_data = base64.b64decode(package['encrypted_data'])
        
        # Verify checksum
        computed_checksum = hashlib.sha256(encrypted_data).hexdigest()
        if computed_checksum != package['checksum']:
            raise ValueError("Data integrity check failed!")
        
        # Decrypt data
        data = self.encryption.decrypt_data(encrypted_data)
        
        logger.info("Data transmission received and verified")
        
        return data, package['metadata']


class PrivacyCompliance:
    """
    Ensures compliance with privacy regulations (GDPR, CCPA, etc.)
    """
    
    def __init__(self):
        """Initialize privacy compliance checker"""
        self.regulations = ['GDPR', 'CCPA', 'HIPAA']
        logger.info("Privacy compliance checker initialized")
    
    def check_gdpr_compliance(self, data_handling: Dict) -> Dict:
        """
        Check GDPR compliance
        
        Args:
            data_handling: Dictionary describing data handling practices
            
        Returns:
            Compliance report
        """
        requirements = {
            'explicit_consent': False,
            'right_to_access': False,
            'right_to_erasure': False,
            'data_minimization': False,
            'purpose_limitation': False,
            'encryption': False
        }
        
        # Check each requirement
        if data_handling.get('consent_obtained'):
            requirements['explicit_consent'] = True
        
        if data_handling.get('user_access_enabled'):
            requirements['right_to_access'] = True
        
        if data_handling.get('deletion_available'):
            requirements['right_to_erasure'] = True
        
        if data_handling.get('minimal_data_collection'):
            requirements['data_minimization'] = True
        
        if data_handling.get('defined_purposes'):
            requirements['purpose_limitation'] = True
        
        if data_handling.get('encryption_enabled'):
            requirements['encryption'] = True
        
        compliance_score = sum(requirements.values()) / len(requirements)
        
        return {
            'regulation': 'GDPR',
            'requirements': requirements,
            'compliance_score': compliance_score,
            'compliant': compliance_score >= 0.8
        }
    
    def generate_compliance_report(self, data_handling: Dict) -> str:
        """
        Generate comprehensive compliance report
        
        Args:
            data_handling: Data handling practices
            
        Returns:
            Compliance report string
        """
        gdpr = self.check_gdpr_compliance(data_handling)
        
        report = []
        report.append("="*60)
        report.append("PRIVACY COMPLIANCE REPORT")
        report.append("="*60)
        report.append(f"\nRegulation: {gdpr['regulation']}")
        report.append(f"Compliance Score: {gdpr['compliance_score']*100:.1f}%")
        report.append(f"Status: {'✓ COMPLIANT' if gdpr['compliant'] else '✗ NON-COMPLIANT'}")
        report.append("\nRequirements:")
        
        for req, status in gdpr['requirements'].items():
            symbol = "✓" if status else "✗"
            report.append(f"  {symbol} {req.replace('_', ' ').title()}")
        
        report.append("="*60)
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    print("Testing Privacy and Security Module...\n")
    
    # Test encryption
    print("1. Testing Encryption:")
    encryptor = NeuralDataEncryption(password="test_password_123")
    test_data = np.random.randn(8, 250)
    
    encrypted = encryptor.encrypt_data(test_data)
    print(f"  Encrypted data size: {len(encrypted)} bytes")
    
    decrypted = encryptor.decrypt_data(encrypted)
    print(f"  Decryption successful: {np.allclose(test_data, decrypted)}")
    
    # Test anonymization
    print("\n2. Testing Anonymization:")
    anonymizer = DataAnonymizer()
    
    original_metadata = {
        'user_id': 'john.doe@example.com',
        'name': 'John Doe',
        'age': 34,
        'timestamp': time.time()
    }
    
    anonymized = anonymizer.anonymize_metadata(original_metadata)
    print(f"  Original: {original_metadata}")
    print(f"  Anonymized: {anonymized}")
    
    # Test consent management
    print("\n3. Testing Consent Management:")
    consent_mgr = ConsentManager()
    
    consent_mgr.record_consent(
        'user_001',
        'data_collection',
        granted=True,
        purposes=['research', 'training']
    )
    
    has_consent = consent_mgr.check_consent('user_001', 'data_collection')
    print(f"  Consent granted: {has_consent}")
    
    summary = consent_mgr.get_consent_summary('user_001')
    print(f"  Consent summary: {json.dumps(summary, indent=2)}")
    
    # Test secure transmission
    print("\n4. Testing Secure Transmission:")
    transmitter = SecureTransmission()
    
    package = transmitter.prepare_for_transmission(test_data, {'trial': 1})
    print(f"  Package prepared with checksum: {package['checksum'][:16]}...")
    
    received_data, received_metadata = transmitter.receive_transmission(package)
    print(f"  Data received successfully: {np.allclose(test_data, received_data)}")
    
    # Test compliance
    print("\n5. Testing Privacy Compliance:")
    compliance = PrivacyCompliance()
    
    data_handling = {
        'consent_obtained': True,
        'user_access_enabled': True,
        'deletion_available': True,
        'minimal_data_collection': True,
        'defined_purposes': True,
        'encryption_enabled': True
    }
    
    report = compliance.generate_compliance_report(data_handling)
    print(report)
    
    print("\nAll tests completed!")