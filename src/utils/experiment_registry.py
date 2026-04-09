"""
Experiment Registry Management

Maintains a registry of all experiments with metadata tracking.
Helps ensure reproducibility and prevents accidental config reuse without documentation.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


class ExperimentRegistry:
    """
    Manages the experiment registry JSON file.
    
    Registry tracks:
    - When experiments were created
    - Their description and purpose
    - Execution status and dates
    - User notes for reproducibility
    """
    
    REGISTRY_FILE = "configs/EXPERIMENT_REGISTRY.json"
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize registry manager."""
        self.project_root = project_root or self._find_project_root()
        self.registry_path = self.project_root / self.REGISTRY_FILE
    
    @staticmethod
    def _find_project_root() -> Path:
        """Find the project root directory (contains configs/ folder)."""
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / 'configs').exists():
                return parent
        return current
    
    def load_registry(self) -> Dict[str, Any]:
        """
        Load the experiment registry.
        
        Returns:
            Dictionary of {experiment_name: {metadata}}
            Returns empty dict if registry doesn't exist yet.
        """
        if not self.registry_path.exists():
            return {}
        
        with open(self.registry_path, 'r') as f:
            return json.load(f)
    
    def save_registry(self, registry: Dict[str, Any]) -> None:
        """
        Save the experiment registry.
        
        Args:
            registry: Dictionary to save as JSON
        """
        # Ensure parent directory exists
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def register_experiment(
        self,
        experiment_name: str,
        description: str,
        notes: str = "",
    ) -> None:
        """
        Register a new experiment or update existing one.
        
        Args:
            experiment_name: Unique experiment identifier
            description: Human-readable description of the experiment
            notes: Optional user notes about the experiment
        
        Raises:
            ValueError: If experiment already exists (to prevent accidental overwrites)
        """
        registry = self.load_registry()
        
        if experiment_name in registry:
            raise ValueError(
                f"Experiment '{experiment_name}' already exists in registry.\n"
                f"Created: {registry[experiment_name]['created_date']}\n"
                f"To reuse a config, create a new experiment name."
            )
        
        registry[experiment_name] = {
            'created_date': datetime.now().isoformat(),
            'description': description,
            'status': 'registered',  # Values: registered, preprocessing, training, completed, failed
            'last_run_date': None,
            'last_run_status': None,
            'notes': notes,
        }
        
        self.save_registry(registry)
    
    def update_run_status(
        self,
        experiment_name: str,
        status: str,
        stage: str = "",
    ) -> None:
        """
        Update the execution status of an experiment.
        
        Args:
            experiment_name: Name of the experiment
            status: Execution status (success, failed, running)
            stage: Pipeline stage (preprocessing, training, evaluation, etc.)
        
        Raises:
            ValueError: If experiment not found in registry
        """
        registry = self.load_registry()
        
        if experiment_name not in registry:
            raise ValueError(
                f"Experiment '{experiment_name}' not found in registry.\n"
                f"Register it first using register_experiment()."
            )
        
        entry = registry[experiment_name]
        entry['last_run_date'] = datetime.now().isoformat()
        entry['last_run_status'] = f"{status} ({stage})" if stage else status
        
        # Update overall status
        if status == 'success' and entry['status'] != 'completed':
            entry['status'] = 'completed'
        elif status == 'failed':
            entry['status'] = 'failed'
        elif status == 'running':
            entry['status'] = 'running'
        
        self.save_registry(registry)
    
    def add_note(self, experiment_name: str, note: str) -> None:
        """
        Add or append notes to an experiment entry.
        
        Args:
            experiment_name: Name of the experiment
            note: Note text to add
        
        Raises:
            ValueError: If experiment not found
        """
        registry = self.load_registry()
        
        if experiment_name not in registry:
            raise ValueError(f"Experiment '{experiment_name}' not found in registry.")
        
        current_notes = registry[experiment_name].get('notes', '')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_note = f"[{timestamp}] {note}"
        if current_notes:
            registry[experiment_name]['notes'] = f"{current_notes}\n{new_note}"
        else:
            registry[experiment_name]['notes'] = new_note
        
        self.save_registry(registry)
    
    def get_experiment_info(self, experiment_name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific experiment.
        
        Args:
            experiment_name: Name of the experiment
        
        Returns:
            Experiment metadata dictionary
        
        Raises:
            ValueError: If experiment not found
        """
        registry = self.load_registry()
        
        if experiment_name not in registry:
            raise ValueError(f"Experiment '{experiment_name}' not found in registry.")
        
        return registry[experiment_name]
    
    def list_experiments(self, status: Optional[str] = None) -> List[str]:
        """
        List all registered experiments, optionally filtered by status.
        
        Args:
            status: Filter by status (registered, completed, failed, running)
        
        Returns:
            List of experiment names
        """
        registry = self.load_registry()
        
        if status is None:
            return list(registry.keys())
        
        return [
            name for name, info in registry.items()
            if info.get('status') == status
        ]
    
    def print_registry(self, status: Optional[str] = None) -> None:
        """
        Pretty-print the registry to console.
        
        Args:
            status: Filter by status
        """
        registry = self.load_registry()
        
        if not registry:
            print("No experiments registered yet.")
            return
        
        print("\nExperiment Registry")
        print("=" * 80)
        
        for exp_name in sorted(registry.keys()):
            info = registry[exp_name]
            
            if status and info.get('status') != status:
                continue
            
            print(f"\n{exp_name}")
            print(f"  Created:  {info.get('created_date', 'N/A')}")
            print(f"  Status:   {info.get('status', 'N/A')}")
            print(f"  Description: {info.get('description', 'N/A')}")
            
            if info.get('last_run_date'):
                print(f"  Last run: {info['last_run_date']}")
                print(f"  Result:   {info.get('last_run_status', 'N/A')}")
            
            if info.get('notes'):
                notes = info['notes']
                lines = notes.split('\n')
                print(f"  Notes:")
                for line in lines:
                    print(f"    {line}")


# Global registry instance
_registry = None


def get_registry(project_root: Optional[Path] = None) -> ExperimentRegistry:
    """
    Get or create the global ExperimentRegistry instance.
    
    Args:
        project_root: Optional project root path
    
    Returns:
        ExperimentRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ExperimentRegistry(project_root)
    return _registry


def register_experiment(
    experiment_name: str,
    description: str,
    notes: str = "",
) -> None:
    """
    Convenience function to register an experiment.
    
    Usage in notebooks:
        from src.utils.experiment_registry import register_experiment
        register_experiment("baseline_exp_001", "RF+SVM baseline on real data")
    
    Args:
        experiment_name: Unique experiment identifier
        description: Human-readable description
        notes: Optional user notes
    """
    get_registry().register_experiment(experiment_name, description, notes)
