package tech.molecules.leet.chem;

import java.util.ArrayList;
import java.util.List;

public class CombinatoricsUtils {


    public static List<Integer> intSeq(int end) {
        return intSeq(0,end,new ArrayList<>());
    }

    public static List<Integer> intSeq(int end, List<Integer> skip) {
        return intSeq(0,end,skip);
    }

    public static List<Integer> intSeq(int start, int end, List<Integer> skip) {
        List<Integer> seq = new ArrayList<>();
        for(int zi=start;zi<end;zi++) {
            if(skip.contains(zi)) {continue;}
            seq.add(zi);
        }
        return seq;
    }

    public static List<List<Integer>> all_permutations(List<Integer> li) {
        if(li.size()==1) {
            ArrayList<List<Integer>> pi = new ArrayList<>();
            pi.add( new ArrayList<>(li) );
            return pi;
        }
        List<List<Integer>> all_permutations = new ArrayList<>();
        for(int zi=0;zi<li.size();zi++) {
            List<Integer> li2 = new ArrayList<>(li);
            Integer xi = li2.get(zi);
            li2.remove(zi);
            List<List<Integer>> perms = all_permutations(li2);
            for(List<Integer> pei : perms) {
                pei.add(xi);
                all_permutations.add(pei);
            }
        }
        return all_permutations;
    }


}
