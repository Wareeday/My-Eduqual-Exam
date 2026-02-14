"""
Neurofeedback and Training System
Implements real-time neurofeedback with gamification elements
"""

import numpy as np
import time
from typing import Dict, List, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeurofeedbackEngine:
    """
    Real-time neurofeedback system for BCI training
    """
    
    def __init__(self, feedback_type: str = 'visual'):
        """
        Initialize neurofeedback engine
        
        Args:
            feedback_type: Type of feedback ('visual', 'auditory', 'haptic')
        """
        self.feedback_type = feedback_type
        self.is_active = False
        self.session_data = []
        
        # Feedback thresholds
        self.thresholds = {
            'excellent': 0.85,
            'good': 0.70,
            'fair': 0.50,
            'poor': 0.30
        }
        
        logger.info(f"Neurofeedback engine initialized ({feedback_type})")
    
    def calculate_performance_score(self, predicted_class: int, 
                                    true_class: int, 
                                    confidence: float) -> float:
        """
        Calculate performance score for current trial
        
        Args:
            predicted_class: Model prediction
            true_class: Ground truth label
            confidence: Prediction confidence
            
        Returns:
            Performance score (0-1)
        """
        # Base score from correctness
        is_correct = (predicted_class == true_class)
        base_score = 1.0 if is_correct else 0.0
        
        # Weighted by confidence
        score = base_score * confidence
        
        return score
    
    def get_feedback_level(self, score: float) -> str:
        """
        Determine feedback level based on score
        
        Args:
            score: Performance score
            
        Returns:
            Feedback level string
        """
        if score >= self.thresholds['excellent']:
            return 'excellent'
        elif score >= self.thresholds['good']:
            return 'good'
        elif score >= self.thresholds['fair']:
            return 'fair'
        elif score >= self.thresholds['poor']:
            return 'poor'
        else:
            return 'needs_improvement'
    
    def provide_visual_feedback(self, level: str):
        """
        Provide visual feedback
        
        Args:
            level: Feedback level
        """
        feedback_colors = {
            'excellent': 'GREEN (Bright)',
            'good': 'GREEN',
            'fair': 'YELLOW',
            'poor': 'ORANGE',
            'needs_improvement': 'RED'
        }
        
        color = feedback_colors.get(level, 'GRAY')
        logger.info(f"[VISUAL FEEDBACK] {color} - {level.upper()}")
        
        # In real implementation, this would update a visual display
        # For example, change color of a progress bar or avatar
    
    def provide_auditory_feedback(self, level: str):
        """
        Provide auditory feedback
        
        Args:
            level: Feedback level
        """
        feedback_sounds = {
            'excellent': 'High pitch beep (positive)',
            'good': 'Medium pitch beep',
            'fair': 'Neutral tone',
            'poor': 'Low pitch beep',
            'needs_improvement': 'Buzzer sound'
        }
        
        sound = feedback_sounds.get(level, 'No sound')
        logger.info(f"[AUDITORY FEEDBACK] {sound}")
        
        # In real implementation, play actual sounds
        # using pygame.mixer or similar
    
    def provide_haptic_feedback(self, level: str):
        """
        Provide haptic feedback
        
        Args:
            level: Feedback level
        """
        feedback_patterns = {
            'excellent': 'Strong vibration (2 pulses)',
            'good': 'Medium vibration (1 pulse)',
            'fair': 'Light vibration',
            'poor': 'Very light vibration',
            'needs_improvement': 'No vibration'
        }
        
        pattern = feedback_patterns.get(level, 'No haptic')
        logger.info(f"[HAPTIC FEEDBACK] {pattern}")
    
    def provide_feedback(self, score: float):
        """
        Provide feedback based on score
        
        Args:
            score: Performance score
        """
        level = self.get_feedback_level(score)
        
        if self.feedback_type == 'visual':
            self.provide_visual_feedback(level)
        elif self.feedback_type == 'auditory':
            self.provide_auditory_feedback(level)
        elif self.feedback_type == 'haptic':
            self.provide_haptic_feedback(level)
        elif self.feedback_type == 'multimodal':
            self.provide_visual_feedback(level)
            self.provide_auditory_feedback(level)
    
    def record_trial(self, trial_data: Dict):
        """
        Record trial data for analysis
        
        Args:
            trial_data: Dictionary with trial information
        """
        trial_data['timestamp'] = time.time()
        self.session_data.append(trial_data)
    
    def get_session_statistics(self) -> Dict:
        """
        Calculate session statistics
        
        Returns:
            Dictionary with session stats
        """
        if not self.session_data:
            return {}
        
        scores = [trial['score'] for trial in self.session_data]
        
        return {
            'total_trials': len(self.session_data),
            'mean_score': np.mean(scores),
            'median_score': np.median(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores)
        }


class GamificationEngine:
    """
    Gamification system to increase engagement and motivation
    """
    
    def __init__(self, player_name: str = "Player"):
        """
        Initialize gamification engine
        
        Args:
            player_name: Player's name
        """
        self.player_name = player_name
        self.level = 1
        self.experience = 0
        self.total_points = 0
        self.achievements = []
        self.streak_days = 0
        self.last_session_date = None
        
        # Experience thresholds for leveling up
        self.level_thresholds = [100, 250, 500, 1000, 2000, 5000, 10000]
        
        # Achievement definitions
        self.achievement_definitions = {
            'first_steps': {
                'name': 'First Steps',
                'description': 'Complete your first training session',
                'condition': lambda stats: stats.get('sessions_completed', 0) >= 1
            },
            'perfect_score': {
                'name': 'Perfect!',
                'description': 'Achieve 100% accuracy in a trial',
                'condition': lambda stats: stats.get('max_score', 0) >= 1.0
            },
            'persistent': {
                'name': 'Persistent Learner',
                'description': 'Train for 7 days in a row',
                'condition': lambda stats: stats.get('streak_days', 0) >= 7
            },
            'master': {
                'name': 'BCI Master',
                'description': 'Reach level 10',
                'condition': lambda stats: stats.get('level', 0) >= 10
            },
            'hundred_trials': {
                'name': 'Century',
                'description': 'Complete 100 trials',
                'condition': lambda stats: stats.get('total_trials', 0) >= 100
            }
        }
        
        logger.info(f"Gamification engine initialized for {player_name}")
    
    def add_experience(self, points: int):
        """
        Add experience points
        
        Args:
            points: Points to add
        """
        self.experience += points
        self.total_points += points
        
        # Check for level up
        if self.check_level_up():
            self.level_up()
    
    def check_level_up(self) -> bool:
        """
        Check if player should level up
        
        Returns:
            True if level up conditions met
        """
        if self.level <= len(self.level_thresholds):
            threshold = self.level_thresholds[self.level - 1]
            return self.experience >= threshold
        return False
    
    def level_up(self):
        """Handle level up"""
        self.level += 1
        self.experience = 0  # Reset experience for next level
        
        logger.info(f"🎉 LEVEL UP! Now at level {self.level}")
        
        # Grant rewards
        self.grant_level_reward()
    
    def grant_level_reward(self):
        """Grant rewards for leveling up"""
        rewards = {
            2: "Unlocked: Advanced Training Mode",
            5: "Unlocked: Custom Feedback Settings",
            10: "Unlocked: Multiplayer Challenges",
            15: "Unlocked: Expert Mode"
        }
        
        if self.level in rewards:
            logger.info(f"🎁 Reward: {rewards[self.level]}")
    
    def calculate_trial_points(self, score: float, difficulty: float = 1.0) -> int:
        """
        Calculate points for a trial
        
        Args:
            score: Trial score (0-1)
            difficulty: Difficulty multiplier
            
        Returns:
            Points earned
        """
        base_points = 10
        bonus_multiplier = 1.0 + (score - 0.5) * 2  # Ranges from 0 to 2
        
        points = int(base_points * bonus_multiplier * difficulty)
        return max(1, points)  # Minimum 1 point
    
    def check_achievements(self, session_stats: Dict):
        """
        Check and unlock achievements
        
        Args:
            session_stats: Current session statistics
        """
        unlocked = []
        
        for achievement_id, achievement in self.achievement_definitions.items():
            if achievement_id not in self.achievements:
                if achievement['condition'](session_stats):
                    self.achievements.append(achievement_id)
                    unlocked.append(achievement)
                    logger.info(f"🏆 Achievement Unlocked: {achievement['name']}")
                    logger.info(f"   {achievement['description']}")
        
        return unlocked
    
    def update_streak(self):
        """Update daily streak"""
        today = time.strftime('%Y-%m-%d')
        
        if self.last_session_date:
            last_date = time.strptime(self.last_session_date, '%Y-%m-%d')
            current_date = time.strptime(today, '%Y-%m-%d')
            
            # Calculate day difference
            diff = (time.mktime(current_date) - time.mktime(last_date)) / 86400
            
            if diff == 1:
                # Consecutive day
                self.streak_days += 1
                logger.info(f"🔥 Streak: {self.streak_days} days!")
            elif diff > 1:
                # Streak broken
                self.streak_days = 1
                logger.info("Streak reset. Keep training!")
        else:
            self.streak_days = 1
        
        self.last_session_date = today
    
    def get_leaderboard_entry(self) -> Dict:
        """
        Get player's leaderboard entry
        
        Returns:
            Dictionary with player stats
        """
        return {
            'player_name': self.player_name,
            'level': self.level,
            'total_points': self.total_points,
            'achievements': len(self.achievements),
            'streak_days': self.streak_days
        }


class AdaptiveDifficultySystem:
    """
    Adjusts task difficulty based on user performance
    """
    
    def __init__(self, initial_difficulty: float = 0.5):
        """
        Initialize adaptive difficulty system
        
        Args:
            initial_difficulty: Starting difficulty (0-1)
        """
        self.difficulty = initial_difficulty
        self.performance_history = []
        self.window_size = 10  # Number of trials to consider
        
        # Difficulty parameters
        self.min_difficulty = 0.1
        self.max_difficulty = 1.0
        self.adjustment_rate = 0.05
        
        # Target performance range
        self.target_min = 0.60
        self.target_max = 0.80
        
        logger.info(f"Adaptive difficulty initialized at {initial_difficulty}")
    
    def update(self, performance_score: float):
        """
        Update difficulty based on performance
        
        Args:
            performance_score: Recent trial performance (0-1)
        """
        self.performance_history.append(performance_score)
        
        # Keep only recent history
        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)
        
        # Calculate average performance
        avg_performance = np.mean(self.performance_history[-self.window_size:])
        
        # Adjust difficulty
        if avg_performance > self.target_max:
            # Too easy, increase difficulty
            self.difficulty = min(
                self.max_difficulty,
                self.difficulty + self.adjustment_rate
            )
            logger.info(f"Increased difficulty to {self.difficulty:.2f}")
            
        elif avg_performance < self.target_min:
            # Too hard, decrease difficulty
            self.difficulty = max(
                self.min_difficulty,
                self.difficulty - self.adjustment_rate
            )
            logger.info(f"Decreased difficulty to {self.difficulty:.2f}")
    
    def get_difficulty(self) -> float:
        """Get current difficulty level"""
        return self.difficulty
    
    def get_difficulty_description(self) -> str:
        """Get human-readable difficulty description"""
        if self.difficulty < 0.3:
            return "Easy"
        elif self.difficulty < 0.6:
            return "Medium"
        elif self.difficulty < 0.9:
            return "Hard"
        else:
            return "Expert"


class TrainingSession:
    """
    Manages a complete training session
    """
    
    def __init__(self, player_name: str = "Player"):
        """
        Initialize training session
        
        Args:
            player_name: Player's name
        """
        self.neurofeedback = NeurofeedbackEngine(feedback_type='multimodal')
        self.gamification = GamificationEngine(player_name)
        self.adaptive_difficulty = AdaptiveDifficultySystem()
        
        self.session_start_time = time.time()
        self.trials_completed = 0
        self.session_scores = []
        
        logger.info(f"Training session started for {player_name}")
    
    def run_trial(self, predicted_class: int, true_class: int, 
                 confidence: float):
        """
        Execute a single training trial
        
        Args:
            predicted_class: Model prediction
            true_class: Ground truth
            confidence: Prediction confidence
        """
        # Calculate performance
        score = self.neurofeedback.calculate_performance_score(
            predicted_class, true_class, confidence
        )
        
        # Provide feedback
        self.neurofeedback.provide_feedback(score)
        
        # Calculate points
        difficulty = self.adaptive_difficulty.get_difficulty()
        points = self.gamification.calculate_trial_points(score, difficulty)
        self.gamification.add_experience(points)
        
        # Update adaptive difficulty
        self.adaptive_difficulty.update(score)
        
        # Record trial
        trial_data = {
            'trial_number': self.trials_completed + 1,
            'predicted_class': predicted_class,
            'true_class': true_class,
            'confidence': confidence,
            'score': score,
            'points': points,
            'difficulty': difficulty
        }
        
        self.neurofeedback.record_trial(trial_data)
        self.session_scores.append(score)
        self.trials_completed += 1
        
        logger.info(f"Trial {self.trials_completed}: Score={score:.2f}, "
                   f"Points={points}, Difficulty={difficulty:.2f}")
    
    def end_session(self):
        """End training session and provide summary"""
        session_duration = time.time() - self.session_start_time
        
        # Get statistics
        stats = self.neurofeedback.get_session_statistics()
        stats['sessions_completed'] = 1
        stats['level'] = self.gamification.level
        stats['streak_days'] = self.gamification.streak_days
        stats['total_trials'] = self.trials_completed
        stats['max_score'] = max(self.session_scores) if self.session_scores else 0
        
        # Update streak
        self.gamification.update_streak()
        
        # Check achievements
        unlocked = self.gamification.check_achievements(stats)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("SESSION SUMMARY")
        logger.info("="*50)
        logger.info(f"Duration: {session_duration/60:.1f} minutes")
        logger.info(f"Trials completed: {self.trials_completed}")
        logger.info(f"Average score: {stats['mean_score']:.2f}")
        logger.info(f"Level: {self.gamification.level}")
        logger.info(f"Total points: {self.gamification.total_points}")
        logger.info(f"Streak: {self.gamification.streak_days} days")
        logger.info(f"Achievements unlocked: {len(unlocked)}")
        logger.info("="*50)
        
        return stats


# Example usage
if __name__ == "__main__":
    print("Testing Neurofeedback and Training System...\n")
    
    # Create training session
    session = TrainingSession(player_name="Test Player")
    
    # Simulate 20 training trials
    print("Running training trials...\n")
    for trial in range(20):
        # Simulate prediction (getting better over time)
        true_class = np.random.randint(0, 4)
        
        # Simulate improving accuracy
        if np.random.random() < 0.6 + (trial * 0.02):
            predicted_class = true_class
            confidence = np.random.uniform(0.7, 0.95)
        else:
            predicted_class = (true_class + 1) % 4
            confidence = np.random.uniform(0.4, 0.7)
        
        # Run trial
        session.run_trial(predicted_class, true_class, confidence)
        
        time.sleep(0.1)  # Small delay for readability
    
    # End session
    print("\n")
    stats = session.end_session()
    
    print("\nAll tests completed!")