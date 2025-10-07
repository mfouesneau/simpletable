# SimpleTable

This library is a refectoring of my SimpleTable data structure to rely on Pandas instead of numpy recarrays.

SimpleTable was born long ago when Pandas did not exist (yes that old! < 2008). It was designed to provide a lot of features to manipulate data arrays in a more human-friendly way. It was also a time when datasets were more humble than what we work with today. It has evolved over time to incorporate the best practices and features of Pandas and Astropy.Table. But it is time to move on to more optimized data structures.

## Features

Two main data structures: `DictDataFrame` and `DataFrame`.

- `DictDataFrame` is primarily a dictionary with flexible and efficient data access and modification (sort, groupby, join, select, filter, etc.)
- `DataFrame` is an extension of the Pandas DataFrame providing a few additional features such as aliases, metadata (e.g. units, descriptions), and input from/output to additional formats (FITS, Ascii/CSV with headers, ECSV).

## What is missing from Pandas and provided here?
- Aliases: Provides a way to create aliases for columns, making it easier to write general codes.
- Metadata: Provides a way to attach and store metadata to datasets and columns (e.g. units, descriptions, comments). Astropy does provide such a mechanism.
- Some input/output formats: Provides a way to read and write data from/to additional formats (FITS, Ascii/CSV with headers, ECSV).

## Why not use Pandas then?

This refactoring is essentially that move: effectively, `class DataFrame(pd.DataFrame)` (i.e. an extension of Pandas DataFrame).

The dictionary part is still remaining to provide a flexible and efficient data access and modification (sort, groupby, join, select, filter, etc.) -- Because you don't always need the full power of Pandas. It's also more lightweight and easier to use for smaller or multidimensional datasets.
