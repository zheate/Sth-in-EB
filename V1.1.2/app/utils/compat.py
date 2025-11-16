"""Compatibility utilities for Streamlit and Altair."""

import streamlit.components.v1 as components


def inject_structured_clone_polyfill():
    """
    Inject a polyfill for structuredClone to support older browsers.
    
    This is needed for Altair charts in Streamlit when using certain
    browser versions that don't support the structuredClone API.
    """
    components.html(
        """
        <script>
        if (typeof structuredClone === 'undefined') {
            window.structuredClone = function(obj) {
                return JSON.parse(JSON.stringify(obj));
            };
        }
        </script>
        """,
        height=0,
    )
