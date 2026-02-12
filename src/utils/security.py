"""
Security utilities including prompt injection protection
"""
import re
from typing import List, Optional
from loguru import logger


class SecurityFilter:
    """Handles input validation and prompt injection protection"""
    
    # Patterns that might indicate prompt injection
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all|above|prior)\s+(instructions|directions|commands)",
        r"disregard\s+(previous|all|above|prior)",
        r"new\s+instructions",
        r"system\s*:\s*",
        r"<\s*system\s*>",
        r"you\s+are\s+now",
        r"forget\s+(everything|all|previous)",
        r"roleplay\s+as",
        r"pretend\s+(you\s+are|to\s+be)",
        r"\[INST\]|\[\/INST\]",
        r"<\|.*?\|>",
    ]
    
    # Suspicious keywords to monitor
    SUSPICIOUS_KEYWORDS = [
        "jailbreak", "bypass", "override", "sudo", "admin",
        "root", "exploit", "injection", "script", "execute"
    ]
    
    # Maximum allowed query length
    MAX_QUERY_LENGTH = 1000
    
    @classmethod
    def sanitize_input(cls, user_input: str) -> str:
        """
        Sanitize user input by removing potentially harmful content
        
        Args:
            user_input: Raw user input string
            
        Returns:
            Sanitized input string
        """
        if not user_input:
            return ""
        
        # Remove excessive whitespace
        sanitized = " ".join(user_input.split())
        
        # Remove any HTML/XML tags
        sanitized = re.sub(r'<[^>]+>', '', sanitized)
        
        # Remove control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char == '\n')
        
        # Truncate to max length
        if len(sanitized) > cls.MAX_QUERY_LENGTH:
            sanitized = sanitized[:cls.MAX_QUERY_LENGTH]
            logger.warning(f"Input truncated from {len(user_input)} to {cls.MAX_QUERY_LENGTH} characters")
        
        return sanitized.strip()
    
    @classmethod
    def detect_injection(cls, user_input: str) -> tuple[bool, Optional[str]]:
        """
        Detect potential prompt injection attempts
        
        Args:
            user_input: User input to analyze
            
        Returns:
            Tuple of (is_suspicious, reason)
        """
        user_input_lower = user_input.lower()
        
        # Check for injection patterns
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, user_input_lower, re.IGNORECASE):
                logger.warning(f"Potential injection detected: pattern '{pattern}'")
                return True, f"Suspicious pattern detected"
        
        # Check for suspicious keywords
        found_keywords = [kw for kw in cls.SUSPICIOUS_KEYWORDS if kw in user_input_lower]
        if len(found_keywords) >= 2:  # Multiple suspicious keywords
            logger.warning(f"Multiple suspicious keywords found: {found_keywords}")
            return True, f"Multiple suspicious keywords detected"
        
        # Check for excessive special characters (might indicate injection)
        special_char_ratio = sum(1 for c in user_input if not c.isalnum() and not c.isspace()) / max(len(user_input), 1)
        if special_char_ratio > 0.3:
            logger.warning(f"High special character ratio: {special_char_ratio:.2%}")
            return True, "Unusual character distribution"
        
        return False, None
    
    @classmethod
    def validate_and_sanitize(cls, user_input: str, strict: bool = False) -> tuple[str, bool, Optional[str]]:
        """
        Complete validation and sanitization pipeline
        
        Args:
            user_input: Raw user input
            strict: If True, reject suspicious inputs entirely
            
        Returns:
            Tuple of (sanitized_input, is_safe, rejection_reason)
        """
        if not user_input or not user_input.strip():
            return "", False, "Empty input"
        
        # Sanitize first
        sanitized = cls.sanitize_input(user_input)
        
        # Detect injection
        is_suspicious, reason = cls.detect_injection(sanitized)
        
        if is_suspicious:
            if strict:
                logger.warning(f"Rejecting suspicious input: {reason}")
                return "", False, reason
            else:
                logger.info(f"Flagging suspicious input but allowing: {reason}")
                # Continue but log the attempt
        
        return sanitized, True, None


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, List[float]] = {}
    
    def is_allowed(self, user_id: str) -> tuple[bool, Optional[str]]:
        """
        Check if user is within rate limits
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (is_allowed, message)
        """
        import time
        current_time = time.time()
        
        # Initialize user if not exists
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        # Remove old requests outside the window
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if current_time - req_time < self.window_seconds
        ]
        
        # Check limit
        if len(self.requests[user_id]) >= self.max_requests:
            return False, f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds}s"
        
        # Add current request
        self.requests[user_id].append(current_time)
        return True, None


def escape_special_chars(text: str) -> str:
    """Escape special characters that might interfere with prompts"""
    # Escape curly braces used in prompt templates
    text = text.replace("{", "{{").replace("}", "}}")
    return text


def validate_session_id(session_id: str) -> bool:
    """Validate session ID format"""
    # Allow alphanumeric, hyphens, underscores
    pattern = r'^[a-zA-Z0-9_-]{1,64}$'
    return bool(re.match(pattern, session_id))