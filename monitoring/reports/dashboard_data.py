"""
Prepare aggregated data for monitoring dashboards.
Creates dashboard-ready tables that can be queried by Lakeview or other BI tools.
"""
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

from monitoring.config import config
from monitoring.utils.databricks_client import DatabricksClient

class DashboardDataPreparation:
    """Prepare and aggregate monitoring data for dashboards"""
    
    def __init__(self, db_client: DatabricksClient):
        self.db = db_client
        self.dashboard_schema = "main.default"
    
    def prepare_quality_trends(self, days_back: int = 30):
        """
        Prepare quality metrics trend data.
        Output: main.default.gdpr_agent_dashboard_quality
        """
        query = f"""
            SELECT 
                DATE(evaluation_date) as date,
                AVG(relevance) as avg_relevance,
                AVG(accuracy) as avg_accuracy,
                AVG(completeness) as avg_completeness,
                AVG(citation) as avg_citation,
                AVG(clarity) as avg_clarity,
                AVG(overall) as avg_overall,
                COUNT(*) as queries_evaluated,
                MIN(overall) as min_overall,
                MAX(overall) as max_overall,
                STDDEV(overall) as std_overall
            FROM {config.QUALITY_METRICS_TABLE}
            WHERE evaluation_date >= current_date() - {days_back}
            GROUP BY DATE(evaluation_date)
            ORDER BY date DESC
        """
        
        df = self.db.query_table(query)
        
        if df.count() > 0:
            # Add metadata
            pdf = df.toPandas()
            pdf['created_at'] = datetime.now()
            pdf['metric_type'] = 'quality'
            
            # Convert back to Spark and save
            spark_df = self.db.spark.createDataFrame(pdf)
            output_table = f"{self.dashboard_schema}.gdpr_agent_dashboard_quality"
            spark_df.write.mode("overwrite").saveAsTable(output_table)
            
            print(f"✅ Quality trends saved to {output_table}")
            return output_table
        else:
            print("⚠️  No quality data available")
            return None
    
    def prepare_performance_trends(self, days_back: int = 30):
        """
        Prepare performance metrics trend data.
        Output: main.default.gdpr_agent_dashboard_performance
        """
        query = f"""
            SELECT 
                p.date,
                COUNT(*) as total_requests,
                COUNT(DISTINCT get_json_object(p.request, '$.dataframe_split.data[0][0]')) as unique_questions,
                AVG((r.timestamp_ms - p.timestamp_ms) / 1000) as avg_latency_seconds,
                PERCENTILE((r.timestamp_ms - p.timestamp_ms) / 1000, 0.5) as p50_latency_seconds,
                PERCENTILE((r.timestamp_ms - p.timestamp_ms) / 1000, 0.95) as p95_latency_seconds,
                PERCENTILE((r.timestamp_ms - p.timestamp_ms) / 1000, 0.99) as p99_latency_seconds,
                MAX((r.timestamp_ms - p.timestamp_ms) / 1000) as max_latency_seconds,
                MIN((r.timestamp_ms - p.timestamp_ms) / 1000) as min_latency_seconds,
                SUM(CASE WHEN p.status_code = 200 THEN 1 ELSE 0 END) as successful_requests,
                SUM(CASE WHEN p.status_code != 200 THEN 1 ELSE 0 END) as failed_requests,
                SUM(CASE WHEN p.status_code = 200 THEN 1 ELSE 0 END) / COUNT(*) * 100 as success_rate_pct
            FROM {config.payload_table} p
            LEFT JOIN {config.response_table} r
                ON p.request_id = r.request_id
            WHERE p.date >= current_date() - {days_back}
            GROUP BY p.date
            ORDER BY p.date DESC
        """
        
        df = self.db.query_table(query)
        
        if df.count() > 0:
            pdf = df.toPandas()
            pdf['created_at'] = datetime.now()
            pdf['metric_type'] = 'performance'
            
            spark_df = self.db.spark.createDataFrame(pdf)
            output_table = f"{self.dashboard_schema}.gdpr_agent_dashboard_performance"
            spark_df.write.mode("overwrite").saveAsTable(output_table)
            
            print(f"✅ Performance trends saved to {output_table}")
            return output_table
        else:
            print("⚠️  No performance data available")
            return None
    
    def prepare_error_summary(self, days_back: int = 30):
        """
        Prepare error summary data.
        Output: main.default.gdpr_agent_dashboard_errors
        """
        query = f"""
            SELECT 
                p.date,
                COUNT(*) as error_count,
                CASE 
                    WHEN p.status_code != 200 THEN 'HTTP_ERROR'
                    WHEN get_json_object(r.response, '$.predictions[0].answer') LIKE '%Error%' THEN 'AGENT_ERROR'
                    WHEN get_json_object(r.response, '$.predictions[0].answer') LIKE '%Exception%' THEN 'AGENT_EXCEPTION'
                    ELSE 'OTHER'
                END as error_type,
                p.status_code,
                COUNT(*) as occurrence_count
            FROM {config.payload_table} p
            LEFT JOIN {config.response_table} r
                ON p.request_id = r.request_id
            WHERE p.date >= current_date() - {days_back}
              AND (
                  p.status_code != 200 
                  OR get_json_object(r.response, '$.predictions[0].answer') LIKE '%Error%'
                  OR get_json_object(r.response, '$.predictions[0].answer') LIKE '%Exception%'
              )
            GROUP BY p.date, error_type, p.status_code
            ORDER BY p.date DESC, occurrence_count DESC
        """
        
        df = self.db.query_table(query)
        
        if df.count() > 0:
            pdf = df.toPandas()
            pdf['created_at'] = datetime.now()
            pdf['metric_type'] = 'errors'
            
            spark_df = self.db.spark.createDataFrame(pdf)
            output_table = f"{self.dashboard_schema}.gdpr_agent_dashboard_errors"
            spark_df.write.mode("overwrite").saveAsTable(output_table)
            
            print(f"✅ Error summary saved to {output_table}")
            return output_table
        else:
            print("✅ No errors to report")
            return None
    
    def prepare_cost_trends(self, days_back: int = 30):
        """
        Prepare cost analysis data.
        Output: main.default.gdpr_agent_dashboard_costs
        """
        query = f"""
            SELECT 
                p.date,
                COUNT(*) as request_count,
                SUM(LENGTH(get_json_object(p.request, '$.dataframe_split.data[0][0]'))) as total_question_chars,
                SUM(LENGTH(get_json_object(r.response, '$.predictions[0].answer'))) as total_answer_chars,
                AVG(LENGTH(get_json_object(p.request, '$.dataframe_split.data[0][0]'))) as avg_question_length,
                AVG(LENGTH(get_json_object(r.response, '$.predictions[0].answer'))) as avg_answer_length
            FROM {config.payload_table} p
            LEFT JOIN {config.response_table} r
                ON p.request_id = r.request_id
            WHERE p.date >= current_date() - {days_back}
              AND p.status_code = 200
            GROUP BY p.date
            ORDER BY p.date DESC
        """
        
        df = self.db.query_table(query)
        
        if df.count() > 0:
            pdf = df.toPandas()
            
            # Calculate estimated tokens and costs
            # Input: question + context (~2500 tokens) + system prompt
            # Output: answer
            pdf['estimated_input_tokens'] = (pdf['total_question_chars'] / 4).astype(int) + (pdf['request_count'] * 650)
            pdf['estimated_output_tokens'] = (pdf['total_answer_chars'] / 4).astype(int)
            pdf['estimated_total_tokens'] = pdf['estimated_input_tokens'] + pdf['estimated_output_tokens']
            
            # Calculate costs using config pricing
            pdf['estimated_input_cost'] = (pdf['estimated_input_tokens'] / 1_000_000) * config.OPENAI_INPUT_COST
            pdf['estimated_output_cost'] = (pdf['estimated_output_tokens'] / 1_000_000) * config.OPENAI_OUTPUT_COST
            pdf['estimated_total_cost'] = pdf['estimated_input_cost'] + pdf['estimated_output_cost']
            pdf['cost_per_request'] = pdf['estimated_total_cost'] / pdf['request_count']
            
            pdf['created_at'] = datetime.now()
            pdf['metric_type'] = 'cost'
            
            spark_df = self.db.spark.createDataFrame(pdf)
            output_table = f"{self.dashboard_schema}.gdpr_agent_dashboard_costs"
            spark_df.write.mode("overwrite").saveAsTable(output_table)
            
            print(f"✅ Cost trends saved to {output_table}")
            return output_table
        else:
            print("⚠️  No cost data available")
            return None
    
    def prepare_query_distribution(self, days_back: int = 30, top_n: int = 50):
        """
        Prepare query distribution and top queries data.
        Output: main.default.gdpr_agent_dashboard_queries
        """
        query = f"""
            SELECT 
                get_json_object(request, '$.dataframe_split.data[0][0]') as question,
                COUNT(*) as frequency,
                MIN(date) as first_seen,
                MAX(date) as last_seen,
                COUNT(DISTINCT date) as days_appeared,
                AVG(LENGTH(get_json_object(request, '$.dataframe_split.data[0][0]'))) as avg_question_length
            FROM {config.payload_table}
            WHERE date >= current_date() - {days_back}
            GROUP BY question
            ORDER BY frequency DESC
            LIMIT {top_n}
        """
        
        df = self.db.query_table(query)
        
        if df.count() > 0:
            pdf = df.toPandas()
            
            # Calculate percentage
            total_requests = pdf['frequency'].sum()
            pdf['percentage'] = (pdf['frequency'] / total_requests * 100).round(2)
            pdf['created_at'] = datetime.now()
            pdf['metric_type'] = 'query_distribution'
            
            spark_df = self.db.spark.createDataFrame(pdf)
            output_table = f"{self.dashboard_schema}.gdpr_agent_dashboard_queries"
            spark_df.write.mode("overwrite").saveAsTable(output_table)
            
            print(f"✅ Query distribution saved to {output_table}")
            return output_table
        else:
            print("⚠️  No query data available")
            return None
    
    def prepare_keyword_trends(self, days_back: int = 30, top_n: int = 50):
        """
        Prepare keyword analysis data.
        Output: main.default.gdpr_agent_dashboard_keywords
        """
        query = f"""
            SELECT 
                keyword,
                COUNT(*) as frequency,
                COUNT(DISTINCT date) as days_appeared,
                MIN(date) as first_seen,
                MAX(date) as last_seen
            FROM (
                SELECT 
                    date,
                    explode(split(lower(get_json_object(request, '$.dataframe_split.data[0][0]')), ' ')) as keyword
                FROM {config.payload_table}
                WHERE date >= current_date() - {days_back}
            )
            WHERE LENGTH(keyword) > 4
              AND keyword NOT RLIKE '[^a-z]'
              AND keyword NOT IN ('what', 'when', 'where', 'which', 'should', 'would', 
                                   'could', 'does', 'have', 'about', 'under', 'their', 
                                   'there', 'these', 'those', 'with', 'from', 'into', 
                                   'that', 'this')
            GROUP BY keyword
            ORDER BY frequency DESC
            LIMIT {top_n}
        """
        
        df = self.db.query_table(query)
        
        if df.count() > 0:
            pdf = df.toPandas()
            pdf['created_at'] = datetime.now()
            pdf['metric_type'] = 'keywords'
            
            spark_df = self.db.spark.createDataFrame(pdf)
            output_table = f"{self.dashboard_schema}.gdpr_agent_dashboard_keywords"
            spark_df.write.mode("overwrite").saveAsTable(output_table)
            
            print(f"✅ Keyword trends saved to {output_table}")
            return output_table
        else:
            print("⚠️  No keyword data available")
            return None
    
    def prepare_summary_kpis(self, days_back: int = 7):
        """
        Prepare high-level KPI summary for dashboard overview.
        Output: main.default.gdpr_agent_dashboard_kpis
        """
        kpis = []
        
        # 1. Quality KPI
        try:
            quality_query = f"""
                SELECT AVG(overall) as avg_quality
                FROM {config.QUALITY_METRICS_TABLE}
                WHERE evaluation_date >= current_date() - {days_back}
            """
            quality_result = self.db.query_table(quality_query).first()
            if quality_result and quality_result.avg_quality:
                kpis.append({
                    'kpi_name': 'Average Quality Score',
                    'kpi_value': float(quality_result.avg_quality),
                    'kpi_unit': '/5',
                    'kpi_category': 'quality',
                    'is_good': quality_result.avg_quality >= config.MIN_QUALITY_SCORE
                })
        except:
            pass
        
        # 2. Performance KPIs
        try:
            perf_query = f"""
                SELECT 
                    COUNT(*) as total_requests,
                    AVG((r.timestamp_ms - p.timestamp_ms) / 1000) as avg_latency,
                    SUM(CASE WHEN p.status_code = 200 THEN 1 ELSE 0 END) / COUNT(*) * 100 as success_rate
                FROM {config.payload_table} p
                LEFT JOIN {config.response_table} r ON p.request_id = r.request_id
                WHERE p.date >= current_date() - {days_back}
            """
            perf_result = self.db.query_table(perf_query).first()
            if perf_result:
                kpis.append({
                    'kpi_name': 'Total Requests',
                    'kpi_value': int(perf_result.total_requests),
                    'kpi_unit': 'requests',
                    'kpi_category': 'volume',
                    'is_good': True
                })
                kpis.append({
                    'kpi_name': 'Average Latency',
                    'kpi_value': float(perf_result.avg_latency),
                    'kpi_unit': 'seconds',
                    'kpi_category': 'performance',
                    'is_good': perf_result.avg_latency <= config.MAX_AVG_LATENCY
                })
                kpis.append({
                    'kpi_name': 'Success Rate',
                    'kpi_value': float(perf_result.success_rate),
                    'kpi_unit': '%',
                    'kpi_category': 'reliability',
                    'is_good': perf_result.success_rate >= config.MIN_SUCCESS_RATE
                })
        except:
            pass
        
        # 3. Cost KPI (if cost metrics exist)
        try:
            cost_query = f"""
                SELECT SUM(estimated_cost_usd) as total_cost
                FROM main.default.gdpr_agent_cost_metrics
                WHERE date >= current_date() - {days_back}
            """
            cost_result = self.db.query_table(cost_query).first()
            if cost_result and cost_result.total_cost:
                kpis.append({
                    'kpi_name': 'Total Cost',
                    'kpi_value': float(cost_result.total_cost),
                    'kpi_unit': 'USD',
                    'kpi_category': 'cost',
                    'is_good': True
                })
        except:
            pass
        
        if kpis:
            kpis_df = pd.DataFrame(kpis)
            kpis_df['period_days'] = days_back
            kpis_df['created_at'] = datetime.now()
            
            spark_df = self.db.spark.createDataFrame(kpis_df)
            output_table = f"{self.dashboard_schema}.gdpr_agent_dashboard_kpis"
            spark_df.write.mode("overwrite").saveAsTable(output_table)
            
            print(f"✅ KPI summary saved to {output_table}")
            return output_table
        else:
            print("⚠️  No KPI data available")
            return None
    
    def prepare_all_dashboard_data(self, days_back: int = 30):
        """
        Prepare all dashboard data tables in one run.
        
        Args:
            days_back: Number of days of historical data to prepare
        
        Returns:
            Dict with table names for each dashboard data type
        """
        print("="*80)
        print(f"📊 PREPARING DASHBOARD DATA (Last {days_back} days)")
        print("="*80)
        
        results = {}
        
        # Prepare each dashboard data table
        print("\n1. Quality Trends...")
        results['quality'] = self.prepare_quality_trends(days_back)
        
        print("\n2. Performance Trends...")
        results['performance'] = self.prepare_performance_trends(days_back)
        
        print("\n3. Error Summary...")
        results['errors'] = self.prepare_error_summary(days_back)
        
        print("\n4. Cost Trends...")
        results['costs'] = self.prepare_cost_trends(days_back)
        
        print("\n5. Query Distribution...")
        results['queries'] = self.prepare_query_distribution(days_back)
        
        print("\n6. Keyword Trends...")
        results['keywords'] = self.prepare_keyword_trends(days_back)
        
        print("\n7. KPI Summary...")
        results['kpis'] = self.prepare_summary_kpis(7)  # Last 7 days for KPIs
        
        print("\n" + "="*80)
        print("✅ Dashboard Data Preparation Complete!")
        print("="*80)
        
        # Print summary
        print("\n📋 Created Dashboard Tables:")
        for data_type, table_name in results.items():
            if table_name:
                print(f"   ✅ {data_type}: {table_name}")
            else:
                print(f"   ⚠️  {data_type}: No data")
        
        return results


def prepare_dashboard_data(days_back: int = 30):
    """
    Convenience function to prepare all dashboard data.
    
    Usage:
        python -m monitoring.reports.dashboard_data
    """
    db_client = DatabricksClient()
    dashboard_prep = DashboardDataPreparation(db_client)
    return dashboard_prep.prepare_all_dashboard_data(days_back)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare dashboard data")
    parser.add_argument("--days-back", type=int, default=30, help="Days of data to prepare")
    
    args = parser.parse_args()
    
    prepare_dashboard_data(args.days_back)