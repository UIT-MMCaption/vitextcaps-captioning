import logging
import os
import shutil

from pycocoevalcap.spice.spice import Spice as _Spice

logger = logging.getLogger(__name__)

# MapDB used by SPICE has a key size limit. Vietnamese captions can exceed it.
# Truncate at word boundary to this many characters.
_MAX_CAPTION_CHARS = 400


def _truncate_caption(text, max_chars=_MAX_CAPTION_CHARS):
    """Truncate a caption to max_chars at a word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to cut at last space to preserve whole words
    last_space = truncated.rfind(' ')
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated


def _truncate_dict(d):
    """Truncate all captions in a gts/res dict (values are lists of strings)."""
    return {
        k: [_truncate_caption(s) for s in v]
        for k, v in d.items()
    }


def _clear_spice_cache():
    """Remove the SPICE MapDB cache to prevent corruption errors."""
    spice_dir = os.path.dirname(os.path.abspath(_Spice.__module__.replace('.', '/')))
    # More reliable: find cache dir relative to the spice.py file
    import pycocoevalcap.spice.spice as spice_mod
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(spice_mod.__file__)), 'cache')
    if os.path.isdir(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            logger.info("Cleared SPICE cache at %s", cache_dir)
        except OSError as e:
            logger.warning("Failed to clear SPICE cache: %s", e)


class Spice:
    def __init__(self):
        self._spice = _Spice()

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys()

        # Truncate long captions to avoid MapDB key size crash
        gts_t = _truncate_dict(gts)
        res_t = _truncate_dict(res)

        # Clear cache before evaluation to prevent corruption
        _clear_spice_cache()

        try:
            score, scores = self._spice.compute_score(gts_t, res_t)
            return score, scores
        except Exception as e:
            logger.warning("SPICE attempt 1 failed: %s. Retrying with fresh cache...", e)
            # Retry once after clearing cache again
            _clear_spice_cache()
            try:
                score, scores = self._spice.compute_score(gts_t, res_t)
                return score, scores
            except Exception as e2:
                logger.error("SPICE attempt 2 failed: %s. Returning 0.", e2)
                return 0.0, [0.0] * len(gts)

    def __str__(self):
        return 'SPICE'
