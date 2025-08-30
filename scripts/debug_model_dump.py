#!/usr/bin/env python3
"""
Debug script to find the exact object causing model_dump error
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def patch_model_dump_calls():
    """Patch all model_dump calls to add debugging"""
    import types
    
    original_getattr = object.__getattribute__
    
    def debug_getattr(obj, name):
        if name == 'model_dump':
            print(f"DEBUG: Accessing model_dump on {type(obj)}")
            print(f"DEBUG: Is dict: {isinstance(obj, dict)}")
            print(f"DEBUG: Object: {repr(obj)[:100]}...")
            
            if isinstance(obj, dict):
                print("ERROR: Trying to call model_dump on dict!")
                import traceback
                traceback.print_stack()
                raise AttributeError(f"dict object has no attribute 'model_dump'")
        
        return original_getattr(obj, name)
    
    # This won't work for built-in dict, but let's try another approach
    pass

def test_with_debug():
    """Test regime detection with debug info"""
    print("Testing regime detection with debugging...")
    
    # Monkey patch hasattr to catch model_dump access
    original_hasattr = hasattr
    
    def debug_hasattr(obj, attr):
        if attr == 'model_dump':
            is_dict = isinstance(obj, dict)
            result = original_hasattr(obj, attr)
            print(f"DEBUG hasattr: {type(obj)} has model_dump? {result} (is_dict: {is_dict})")
            if is_dict and result:
                print(f"WARNING: Dict has model_dump attribute! Dict keys: {list(obj.keys()) if obj else 'empty'}")
        return original_hasattr(obj, attr)
    
    # Replace hasattr globally
    import builtins
    builtins.hasattr = debug_hasattr
    
    try:
        from agents.regime_detector import get_current_market_regime
        
        print("Calling get_current_market_regime...")
        regime = get_current_market_regime('2023-01-02')
        print(f"Result: {regime}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original hasattr
        builtins.hasattr = original_hasattr

if __name__ == "__main__":
    test_with_debug()