package com.github.kevindrosendahl.javaannbench.dataset;

import com.google.common.base.Preconditions;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CsvVectorLoader {

  private static final Logger LOGGER = LoggerFactory.getLogger(CsvVectorLoader.class);

  public static List<float[]> loadVectors(Path path, int dimensions) throws IOException {
    LOGGER.info("loading vectors from {}", path);

    var vectors = new ArrayList<float[]>();

    try (var reader = Files.newBufferedReader(path); var parser = new CSVParser(reader,
        CSVFormat.DEFAULT)) {
      for (var record : parser) {
        Preconditions.checkArgument(record.size() == dimensions,
            "row's dimensions %s does not match expected %s", record.size(), dimensions);
        var vector = new float[record.size()];

        var dimension = 0;
        for (var value : record) {
          vector[dimension++] = Float.parseFloat(value);
        }

        vectors.add(vector);
      }
    }

    LOGGER.info("finished loading vectors from {}", path);
    return vectors;
  }

  public static List<List<Integer>> loadGroundTruth(Path path) throws IOException {
    LOGGER.info("loading ground truths from {}", path);

    var groundTruths = new ArrayList<List<Integer>>();

    try (var reader = Files.newBufferedReader(path); var parser = new CSVParser(reader,
        CSVFormat.DEFAULT)) {
      for (var record : parser) {
        var groundTruth = new ArrayList<Integer>(record.size());
        for (var value : record) {
          groundTruth.add(Integer.parseInt(value));
        }

        groundTruths.add(groundTruth);
      }
    }

    LOGGER.info("finished loading ground truths from {}", path);
    return groundTruths;
  }
}
