import importlib
class UtilModule:
    @staticmethod
    def load_class_from_file(filepath, classname, *args, **kwargs):
        spec = importlib.util.spec_from_file_location("module.name", filepath)
        
        module = importlib.util.module_from_spec(spec)
        
        spec.loader.exec_module(module)
        
        cls = getattr(module, classname)
        return cls(*args, **kwargs)
