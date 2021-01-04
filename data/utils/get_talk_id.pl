#!/usr/bin/perl
use strict;
use warnings;
 
my $file = $ARGV[0] or die "Need to get CSV file on the command line\n";
 open(FH, '>', "verenay2.csv") or die $!;
my $index = 0;
open(my $data, '<', $file) or die "Could not open '$file' $!\n";
 my  $first_line = <$data>;
 print FH $first_line."talk_id\n";
while (my $line = <$data>) {
  chomp $line;
 
  my @fields = split "," , $line;
  my $url =  $fields[-1];
  my $curl = `curl $url`;
  my $talk_id = "-";
  if( $curl =~ m/content\=\"ted\:\/\/talks\/(\d+)/){
  #if( $curl =~ m/"talk_id":(\d+)/){
        $talk_id = $1;
print FH $line.",$talk_id\n";
  }
  else{
print FH $line.",$talk_id\n";
 
  }
 
  $index++;
  print $index;
}
