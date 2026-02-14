from typing import Dict, List, Optional
from datetime import datetime
import re

class RuleEngine:
    """Rule-based detection engine for SOC baseline."""
    
    def __init__(self):
        # Configuration thresholds
        self.failed_login_threshold = 5
        self.failed_login_window = 60  # seconds
        self.brute_force_threshold = 10
        self.suspicious_ips = set()
        
        # State tracking
        self.failed_logins = {}  # ip -> list of timestamps
        
    def check_rules(self, log_text: str, metadata: Dict) -> Dict:
        """
        Check all rules against the log.
        Returns:
            dict: {
                'rule_based_alert': bool,
                'rule_reason': str | None,
                'rule_id': str | None
            }
        """
        # Failed login burst detection
        if self._check_failed_login(log_text, metadata):
            return {
                'rule_based_alert': True,
                'rule_reason': 'Failed Login Burst Detected',
                'rule_id': 'R001'
            }
            
        # Privilege escalation keyword
        if self._check_privilege_escalation(log_text):
            return {
                'rule_based_alert': True,
                'rule_reason': 'Potential Privilege Escalation',
                'rule_id': 'R002'
            }
            
        # Suspicious IP
        if self._check_suspicious_ip(metadata.get('ip_address')):
             return {
                'rule_based_alert': True,
                'rule_reason': 'Known Suspicious IP Activity',
                'rule_id': 'R003'
            }
            
        return {'rule_based_alert': False, 'rule_reason': None, 'rule_id': None}

    def _check_failed_login(self, log_text: str, metadata: Dict) -> bool:
        """Detect burst of failed logins from same IP."""
        if "failed" in log_text.lower() and "login" in log_text.lower():
            ip = metadata.get('ip_address')
            if not ip:
                return False
                
            now = datetime.now().timestamp()
            if ip not in self.failed_logins:
                self.failed_logins[ip] = []
            
            # Add current timestamp
            self.failed_logins[ip].append(now)
            
            # Prune old timestamps
            self.failed_logins[ip] = [
                t for t in self.failed_logins[ip] 
                if now - t <= self.failed_login_window
            ]
            
            if len(self.failed_logins[ip]) >= self.failed_login_threshold:
                return True
                
        return False

    def _check_privilege_escalation(self, log_text: str) -> bool:
        """Check for sudo/root access patterns."""
        keywords = ["sudo", "su root", "uid=0", "gid=0", "privilege escalation"]
        return any(k in log_text.lower() for k in keywords)

    def _check_suspicious_ip(self, ip: str) -> bool:
        """Check against known suspicious IPs."""
        return ip in self.suspicious_ips

    def add_suspicious_ip(self, ip: str):
        self.suspicious_ips.add(ip)
