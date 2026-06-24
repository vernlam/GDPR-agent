"""
Reporting modules for monitoring data.

Includes:
- DashboardDataPreparation: Prepare aggregated data for Lakeview dashboards
- prepare_dashboard_data: Convenience function to prepare all dashboard tables
"""

from monitoring.reports.dashboard_data import (
    DashboardDataPreparation,
    prepare_dashboard_data
)

__all__ = [
    "DashboardDataPreparation",
    "prepare_dashboard_data",
]