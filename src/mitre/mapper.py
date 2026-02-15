from typing import Dict, Optional

class MitreMapper:
    """Maps detection outputs to MITRE ATT&CK framework."""
    
    def __init__(self):
        # Technique ID -> Tactic, Name
        self.technique_map = {
            'T1110': ('Credential Access', 'Brute Force'),
            'T1078': ('Defense Evasion', 'Valid Accounts'),
            'T1059': ('Execution', 'Command and Scripting Interpreter'),
            'T1046': ('Discovery', 'Network Service Discovery'),
            'T1003': ('Credential Access', 'OS Credential Dumping')
        }
        
        # Rule ID -> Technique ID
        self.rule_mapping = {
            'R001': 'T1110',  # Failed Login Burst
            'R002': 'T1078',  # Privilege Ecalation (close enough mapping)
            'R003': 'T1046'   # Suspicious IP scanning
        }
        
    def map_alert(self, rule_id: Optional[str] = None, log_text: str = "") -> Dict:
        """
        Map a rule ID or log content to MITRE TTPs.
        """
        technique_id = None
        
        if rule_id and rule_id in self.rule_mapping:
            technique_id = self.rule_mapping[rule_id]
        else:
            # Fallback to simple keyword mapping
            if "brute" in log_text.lower():
                technique_id = 'T1110'
            elif "sudo" in log_text.lower():
                technique_id = 'T1078'
            elif "cmd" in log_text.lower() or "powershell" in log_text.lower():
                technique_id = 'T1059'
                
        if technique_id:
            tactic, name = self.technique_map.get(technique_id, ('Unknown', 'Unknown'))
            return {
                'mitre_technique_id': technique_id,
                'mitre_technique_name': name,
                'mitre_tactic': tactic
            }
            
        return {
            'mitre_technique_id': None,
            'mitre_technique_name': None,
            'mitre_tactic': None
        }
