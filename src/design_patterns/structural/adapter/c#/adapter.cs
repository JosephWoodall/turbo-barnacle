using System;
using System.Collections.Generic;

// Existing class that fetches data from a database
class DatabaseFetcher {
    public List<Dictionary<string, string>> FetchData() {
        // fetches data from the database and returns it as a list of dictionaries
        return new List<Dictionary<string, string>>();
    }
}

// Interface for fetching data from a specific table
interface ITableFetcher {
    List<Dictionary<string, string>> FetchTableData();
}

// Adapter class that fetches data from a specific table using the DatabaseFetcher
class TableFetcherAdapter : ITableFetcher {
    private readonly DatabaseFetcher _databaseFetcher;

    public TableFetcherAdapter(DatabaseFetcher databaseFetcher) {
        _databaseFetcher = databaseFetcher;
    }

    public List<Dictionary<string, string>> FetchTableData() {
        List<Dictionary<string, string>> data = _databaseFetcher.FetchData();
        // filter the data to only include rows from the specific table
        List<Dictionary<string, string>> filteredData = FilterData(data);
        return filteredData;
    }

    private List<Dictionary<string, string>> FilterData(List<Dictionary<string, string>> data) {
        // filter the data to only include rows from the specific table
        return new List<Dictionary<string, string>>();
    }
}

// Example usage
class Program {
    static void Main(string[] args) {
        DatabaseFetcher databaseFetcher = new DatabaseFetcher();
        TableFetcherAdapter adapter = new TableFetcherAdapter(databaseFetcher);

        List<Dictionary<string, string>> tableData = adapter.FetchTableData();
        Console.WriteLine(tableData);
    }
}
