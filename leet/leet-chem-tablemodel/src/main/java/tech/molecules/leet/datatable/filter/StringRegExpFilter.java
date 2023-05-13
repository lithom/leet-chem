package tech.molecules.leet.datatable.filter;

import tech.molecules.leet.datatable.*;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class StringRegExpFilter extends AbstractCachedDataFilter<String> {

    public static StringRegExpFilterType TYPE = new StringRegExpFilterType();

    public static class StringRegExpFilterType implements DataFilterType<String> {
        @Override
        public String getFilterName() {
            return "StringRegExpFilter";
        }

        @Override
        public boolean requiresInitialization() {
            return true;
        }

        @Override
        public DataFilter<String> createInstance(DataTableColumn<?,String> column) {
            return new StringRegExpFilter();
        }

    }

    /**
     * Configuration
     */
    private String regexp;
    private Pattern pattern;


    public void setRegExp(String regexp) {
        this.regexp = regexp;
        this.pattern = Pattern.compile(this.regexp);
        fireFilterChanged();
    }

    @Override
    public boolean filterRow(String vi) {
        //try {Thread.sleep(50);
        //} catch (InterruptedException e) { throw new RuntimeException(e);}

        return  ! this.pattern.matcher(vi).find();
    }

    @Override
    public DataFilterType<String> getDataFilterType() {
        return StringRegExpFilter.TYPE;
    }

    @Override
    public double getApproximateFilterSpeed() {
        return 0.35;
    }
}
