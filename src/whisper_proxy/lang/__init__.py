"""Language-name → ISO 639-1 alpha-2 lookup."""

from .map import alpha2_to_openarc_language, name_to_alpha2, normalize_name

__all__ = ["alpha2_to_openarc_language", "name_to_alpha2", "normalize_name"]
