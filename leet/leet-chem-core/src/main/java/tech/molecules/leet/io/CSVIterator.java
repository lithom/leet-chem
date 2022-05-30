package tech.molecules.leet.io;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class CSVIterator implements Iterator<List<String>> {
    String file;
    BufferedReader in;
    boolean skip_first;
    List<Integer> col_idx;

    boolean next_line_fetched = false;
    String last_line = null;
    List<String> last_extracted = new ArrayList<>();

    int idx;

    public CSVIterator(String file, boolean skip_first, List<Integer> col_idx) throws IOException {
        this.file = file;
        this.skip_first = skip_first;
        this.col_idx = col_idx;
        in = new BufferedReader(new FileReader(this.file));
        init();
    }

    public CSVIterator(BufferedReader in, boolean skip_first, List<Integer> col_idx) throws IOException {
        this.in = in;
        this.file = file;
        this.skip_first = skip_first;
        this.col_idx = col_idx;
        init();
    }

    private void init() throws IOException {
        if (this.skip_first) {
            last_line = in.readLine();
        } else {
            last_line = "";
        }
        this.next_line_fetched = false;
    }

    @Override
    public boolean hasNext() {
        if (!next_line_fetched) {
            //this.next_line_fetched =
            fetchNextLine();
        }
        return last_line != null;
    }

    private boolean fetchNextLine() {
        try {
            last_line = in.readLine();
            if (last_line == null) {
                return false;
            }
            // extract:
            String splits[] = last_line.split(",");
            last_extracted = new ArrayList<>();
            for (int zi = 0; zi < col_idx.size(); zi++) {
                last_extracted.add(splits[this.col_idx.get(zi)]);
            }

        } catch (IOException e) {
            System.out.println("[WARN] IOException in CSVIterator..");
            last_line = null;
            return false;
        }
        return true;
    }

    @Override
    public List<String> next() {
        if (!next_line_fetched) {
            fetchNextLine();
        }
        next_line_fetched = false;
        return last_extracted;
    }
}
