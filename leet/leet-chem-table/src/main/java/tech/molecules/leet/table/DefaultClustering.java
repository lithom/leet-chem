package tech.molecules.leet.table;

import java.awt.*;
import java.util.*;
import java.util.List;

public class DefaultClustering<T> implements NClassification {

    public static class DefaultClass implements NClassification.NClass {
        private String name;
        private String description;
        private Set<String> members = new HashSet<>();
        private Color color;

        public DefaultClass(String name, String description, Set<String> members, Color color) {
            this.name = name;
            this.description = description;
            this.members = members;
            this.color = color;
        }

        @Override
        public String getName() {
            return this.name;
        }

        @Override
        public String getDescription() {
            return this.description;
        }

        @Override
        public Color getColor() {
            return this.color;
        }
        @Override
        public boolean isMember(String ki) {
            return members.contains(ki);
        }
        @Override
        public Set<String> getMembers() {
            return Collections.unmodifiableSet(this.members);
        }

        @Override
        public void setColor(Color c) {
            this.color = c;
        }
    }

    private List<NClass> classes = new ArrayList<>();

    @Override
    public List<NClass> getClasses() {
        return null;
    }

    @Override
    public void addClass(NClass ci) {
        this.classes.add(ci);
    }

    @Override
    public void removeClass(NClass ci) {
        this.classes.remove(ci);
    }

    private void fireClassChanged(NClass ci) {
        for(ClassificationListener li : listeners) {li.classChanged(ci);}
        fireClassificationChanged();
    }

    private void fireClassificationChanged() {
        for(ClassificationListener li : listeners) {li.classificationChanged();}
    }
    private List<ClassificationListener> listeners = new ArrayList<>();

    @Override
    public List<NClass> getClassesForRow(String rowid) {
        List<NClass> classes = new ArrayList<>();
        for(NClass ci : this.classes) { if(ci.isMember(rowid)) {classes.add(ci);}}
        return classes;
    }

    @Override
    public void addClassificationListener(ClassificationListener li) {
        listeners.add(li);
    }

    @Override
    public void removeClassificationListener(ClassificationListener li) {
        listeners.remove(li);
    }
}
