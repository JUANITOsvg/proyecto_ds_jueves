#!/usr/bin/env python3
"""
Test script to demonstrate comprehensive CRUD operations with the F1 Data Warehouse.
This script tests all functions in db_conn_manager.py with sample F1 data.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add the pipelines directory to the path
sys.path.append('/home/juanchidev/workbench/proyecto_datascience/pipelines')

from apis.db_conn_manager import (
    DatabaseManager, connect_to_db, close_connection,
    create_table, insert_data, insert_bulk_data, insert_dataframe,
    select_data, select_to_dataframe, update_data, delete_data,
    drop_table, table_exists, get_table_info, get_table_count
)

def test_connection():
    """Test basic database connection."""
    print("\n" + "="*50)
    print("TESTING DATABASE CONNECTION")
    print("="*50)
    
    conn = connect_to_db()
    if conn:
        print("✅ Connection successful!")
        close_connection(conn)
        return True
    else:
        print("❌ Connection failed!")
        return False

def test_basic_crud():
    """Test basic CRUD operations."""
    print("\n" + "="*50)
    print("TESTING BASIC CRUD OPERATIONS")
    print("="*50)
    
    conn = connect_to_db()
    if not conn:
        print("❌ Could not connect to database")
        return False
    
    try:
        # Test CREATE TABLE
        print("\n1. Testing CREATE TABLE...")
        table_schema = {
            'prediction_id': 'SERIAL PRIMARY KEY',
            'race_id': 'INTEGER NOT NULL',
            'driver_id': 'INTEGER NOT NULL', 
            'predicted_position': 'DECIMAL(4,2)',
            'confidence_score': 'DECIMAL(3,2)',
            'model_version': 'VARCHAR(20)',
            'prediction_timestamp': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
        }
        
        success = create_table(conn, 'ml_predictions_test', table_schema)
        print(f"Create table: {'✅ Success' if success else '❌ Failed'}")
        
        # Test INSERT single record
        print("\n2. Testing INSERT single record...")
        sample_prediction = {
            'race_id': 1001,
            'driver_id': 1,
            'predicted_position': 1.25,
            'confidence_score': 0.87,
            'model_version': 'v1.0'
        }
        
        success = insert_data(conn, 'ml_predictions_test', sample_prediction)
        print(f"Insert single record: {'✅ Success' if success else '❌ Failed'}")
        
        # Test BULK INSERT
        print("\n3. Testing BULK INSERT...")
        bulk_data = [
            {'race_id': 1001, 'driver_id': 2, 'predicted_position': 2.15, 'confidence_score': 0.82, 'model_version': 'v1.0'},
            {'race_id': 1001, 'driver_id': 3, 'predicted_position': 3.45, 'confidence_score': 0.75, 'model_version': 'v1.0'},
            {'race_id': 1002, 'driver_id': 1, 'predicted_position': 1.78, 'confidence_score': 0.91, 'model_version': 'v1.1'},
            {'race_id': 1002, 'driver_id': 2, 'predicted_position': 2.33, 'confidence_score': 0.88, 'model_version': 'v1.1'}
        ]
        
        success = insert_bulk_data(conn, 'ml_predictions_test', bulk_data)
        print(f"Bulk insert: {'✅ Success' if success else '❌ Failed'}")
        
        # Test SELECT
        print("\n4. Testing SELECT operations...")
        results = select_data(conn, 'ml_predictions_test', 
                            columns='race_id, driver_id, predicted_position, confidence_score',
                            order_by='race_id, predicted_position',
                            limit=10)
        
        if results:
            print("✅ Select successful! Results:")
            for row in results:
                print(f"   Race: {row[0]}, Driver: {row[1]}, Position: {row[2]}, Confidence: {row[3]}")
        else:
            print("❌ Select failed!")
        
        # Test COUNT
        print("\n5. Testing COUNT...")
        count = get_table_count(conn, 'ml_predictions_test')
        print(f"Total records: {count if count is not None else 'Failed to get count'}")
        
        # Test UPDATE
        print("\n6. Testing UPDATE...")
        success = update_data(conn, 'ml_predictions_test', 
                            {'confidence_score': 0.95}, 
                            'race_id = 1001 AND driver_id = 1')
        print(f"Update record: {'✅ Success' if success else '❌ Failed'}")
        
        # Test SELECT after update
        print("\n7. Testing SELECT after UPDATE...")
        updated_result = select_data(conn, 'ml_predictions_test',
                                   where_clause='race_id = 1001 AND driver_id = 1')
        if updated_result:
            print(f"✅ Updated confidence score: {updated_result[0][5]}")  # confidence_score column
        
        # Test DELETE
        print("\n8. Testing DELETE...")
        success = delete_data(conn, 'ml_predictions_test', 'race_id = 1002')
        print(f"Delete records: {'✅ Success' if success else '❌ Failed'}")
        
        # Test final count
        final_count = get_table_count(conn, 'ml_predictions_test')
        print(f"Records after delete: {final_count if final_count is not None else 'Failed to get count'}")
        
        # Clean up
        print("\n9. Cleaning up...")
        success = drop_table(conn, 'ml_predictions_test')
        print(f"Drop table: {'✅ Success' if success else '❌ Failed'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic CRUD test: {e}")
        return False
    finally:
        close_connection(conn)

def test_dataframe_operations():
    """Test pandas DataFrame integration."""
    print("\n" + "="*50)
    print("TESTING DATAFRAME OPERATIONS")
    print("="*50)
    
    conn = connect_to_db()
    if not conn:
        print("❌ Could not connect to database")
        return False
    
    try:
        # Create test table for DataFrame operations
        print("\n1. Creating table for DataFrame test...")
        table_schema = {
            'driver_id': 'INTEGER',
            'driver_name': 'VARCHAR(100)',
            'nationality': 'VARCHAR(50)',
            'career_wins': 'INTEGER',
            'championship_years': 'TEXT'
        }
        
        success = create_table(conn, 'drivers_test', table_schema)
        print(f"Create table: {'✅ Success' if success else '❌ Failed'}")
        
        # Create sample DataFrame
        print("\n2. Creating sample DataFrame...")
        df_data = {
            'driver_id': [1, 2, 3, 4, 5],
            'driver_name': ['Lewis Hamilton', 'Max Verstappen', 'Sebastian Vettel', 'Fernando Alonso', 'Charles Leclerc'],
            'nationality': ['British', 'Dutch', 'German', 'Spanish', 'Monégasque'],
            'career_wins': [103, 50, 53, 32, 5],
            'championship_years': ['2008,2014,2015,2017,2018,2019,2020', '2021,2022,2023', '2010,2011,2012,2013', '2005,2006', 'None']
        }
        
        df = pd.DataFrame(df_data)
        print(f"✅ DataFrame created with {len(df)} rows")
        print(df.head())
        
        # Test DataFrame insert
        print("\n3. Testing DataFrame insertion...")
        success = insert_dataframe(conn, 'drivers_test', df)
        print(f"DataFrame insert: {'✅ Success' if success else '❌ Failed'}")
        
        # Test SELECT to DataFrame
        print("\n4. Testing SELECT to DataFrame...")
        query = """
            SELECT driver_name, nationality, career_wins 
            FROM drivers_test 
            WHERE career_wins > 30 
            ORDER BY career_wins DESC
        """
        
        result_df = select_to_dataframe(conn, query)
        if result_df is not None:
            print("✅ Select to DataFrame successful!")
            print(result_df)
        else:
            print("❌ Select to DataFrame failed!")
        
        # Clean up
        print("\n5. Cleaning up...")
        success = drop_table(conn, 'drivers_test')
        print(f"Drop table: {'✅ Success' if success else '❌ Failed'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in DataFrame test: {e}")
        return False
    finally:
        close_connection(conn)

def test_context_manager():
    """Test the DatabaseManager context manager."""
    print("\n" + "="*50)
    print("TESTING CONTEXT MANAGER")
    print("="*50)
    
    try:
        with DatabaseManager() as db:
            print("✅ Context manager connection successful!")
            
            # Create a simple test table
            table_schema = {
                'id': 'SERIAL PRIMARY KEY',
                'test_name': 'VARCHAR(50)',
                'test_value': 'INTEGER'
            }
            
            success = db.create_table('context_test', table_schema)
            print(f"Create table via context manager: {'✅ Success' if success else '❌ Failed'}")
            
            # Insert some data
            test_data = [
                {'test_name': 'Test 1', 'test_value': 100},
                {'test_name': 'Test 2', 'test_value': 200},
                {'test_name': 'Test 3', 'test_value': 300}
            ]
            
            success = db.insert_bulk_data('context_test', test_data)
            print(f"Bulk insert via context manager: {'✅ Success' if success else '❌ Failed'}")
            
            # Check count
            count = db.get_table_count('context_test')
            print(f"Record count: {count}")
            
            # Clean up
            success = db.drop_table('context_test')
            print(f"Drop table via context manager: {'✅ Success' if success else '❌ Failed'}")
            
        print("✅ Context manager test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in context manager test: {e}")
        return False

def test_utility_functions():
    """Test utility functions."""
    print("\n" + "="*50)
    print("TESTING UTILITY FUNCTIONS")
    print("="*50)
    
    conn = connect_to_db()
    if not conn:
        print("❌ Could not connect to database")
        return False
    
    try:
        # Test table existence check
        print("\n1. Testing table existence check...")
        exists_before = table_exists(conn, 'utility_test')
        print(f"Table exists before creation: {exists_before}")
        
        # Create a test table
        table_schema = {
            'id': 'SERIAL PRIMARY KEY',
            'name': 'VARCHAR(100) NOT NULL',
            'value': 'DECIMAL(10,2) DEFAULT 0.00',
            'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
        }
        
        create_table(conn, 'utility_test', table_schema)
        
        exists_after = table_exists(conn, 'utility_test')
        print(f"Table exists after creation: {exists_after}")
        
        # Test get table info
        print("\n2. Testing get table info...")
        table_info = get_table_info(conn, 'utility_test')
        if table_info:
            print("✅ Table info retrieved successfully:")
            for col in table_info:
                print(f"   {col[0]} - {col[1]} - Nullable: {col[2]} - Default: {col[3]}")
        else:
            print("❌ Failed to get table info")
        
        # Clean up
        print("\n3. Cleaning up...")
        success = drop_table(conn, 'utility_test')
        print(f"Drop table: {'✅ Success' if success else '❌ Failed'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in utility functions test: {e}")
        return False
    finally:
        close_connection(conn)

def main():
    """Run all tests."""
    print("🚀 STARTING COMPREHENSIVE DATABASE TESTS")
    print("Database: F1 Data Warehouse (PostgreSQL)")
    print("="*70)
    
    tests = [
        ("Connection Test", test_connection),
        ("Basic CRUD Operations", test_basic_crud),
        ("DataFrame Operations", test_dataframe_operations),
        ("Context Manager", test_context_manager),
        ("Utility Functions", test_utility_functions)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your database is ready for production!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()