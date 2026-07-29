import pytest
from tools.toollens.music import MusicTools

@pytest.fixture
def music_instance():
    config = {
        'videos': {
            'youtube_24h': ['dQw4w9WgXcQ', '3tmd-ClpJxA'],
            'youtube_trending_overall': ['9bZkp7q19f0'],
            'youtube_weekly': []
        },
        'charts': {
            'Artist_100': [{'rank': 1, 'artist': 'Taylor Swift', 'weeks': 10}],
            'BILLBOARD_200': [{'rank': 1, 'album': 'Midnights', 'artist': 'Taylor Swift', 'last_week': 1}],
            'Billboard_200': [{'rank': 1, 'album': 'Midnights', 'artist': 'Taylor Swift'}],
            'Billboard_Global_Excl_US': [],
            'Billboard_Hot_100': [{'rank': 1, 'song': 'Anti-Hero', 'artist': 'Taylor Swift'}],
            'Catalog_Albums': [{'rank': 1, 'album': 'Thriller', 'artist': 'Michael Jackson'}],
            'Greatest_of_All_Time_Hot_100_Songs': [{'rank': 1, 'song': 'Blinding Lights', 'artist': 'The Weeknd'}],
            'Independent_Albums': [{'rank': 1, 'album': 'RTJ4', 'artist': 'Run The Jewels'}],
            'Year_End_Billboard_Global_200': [{'rank': 1, 'song': 'Heat Waves', 'artist': 'Glass Animals'}],
            'Year_End_Top_Artists': [{'rank': 1, 'artist': 'Taylor Swift'}]
        },
        'users': {
            'spotify_user_1': {'display_name': 'John Doe', 'id': 'spotify_user_1', 'followers': 100},
            'unknown_user': {}
        },
        'albums': {
            'List_User_Albums': {
                'spotify_user_1': ['album1', 'album2'],
                'empty_user': []
            },
            'New_releases': {
                'US': [{'album': 'Album1', 'artist': 'Artist1'}],
                'GB': []
            }
        },
        'search': {
            'auto_complete': {
                'hello': ['hello world', 'hello goodbye'],
                'empty': []
            },
            'boy_groups': ['BTS', 'EXO'],
            'girl_groups': ['BLACKPINK', 'TWICE'],
            'random_boy_group': 'BTS',
            'random_song': {
                'artist': 'Taylor Swift',
                'album': '1989',
                'song': 'Shake It Off'
            }
        },
        'channels': ['channel1', 'channel2']
    }
    return MusicTools(initial_config=config)

# ---------- youtube_24h ----------
def test_youtube_24h_normal(music_instance):
    result = music_instance.youtube_24h()
    assert isinstance(result, dict)
    assert 'videos' in result
    assert len(result['videos']) == 2

def test_youtube_24h_empty(music_instance):
    """Test when config has empty list (edge case)"""
    # The fixture has non-empty data; override by creating instance with empty config
    empty_config = {'videos': {'youtube_24h': []}}
    obj = MusicTools(initial_config=empty_config)
    result = obj.youtube_24h()
    assert isinstance(result, dict)
    assert result['videos'] == []

# ---------- youtube_trending_overall ----------
def test_youtube_trending_overall_normal(music_instance):
    result = music_instance.youtube_trending_overall()
    assert isinstance(result, dict)
    assert 'videos' in result
    assert len(result['videos']) == 1

def test_youtube_trending_overall_missing_config(music_instance):
    """Test when no config key exists"""
    obj = MusicTools(initial_config={})
    result = obj.youtube_trending_overall()
    assert isinstance(result, dict)
    assert 'error' in result

# ---------- youtube_weekly ----------
def test_youtube_weekly_normal(music_instance):
    result = music_instance.youtube_weekly()
    assert isinstance(result, dict)
    assert 'videos' in result
    assert result['videos'] == []

# ---------- Artist_100 ----------
def test_artist_100_normal(music_instance):
    result = music_instance.Artist_100(date='2023-01-01')
    assert isinstance(result, dict)
    assert 'chart' in result
    assert len(result['chart']) == 1

def test_artist_100_invalid_date(music_instance):
    """Test with empty string date (edge case)"""
    result = music_instance.Artist_100(date='')
    assert isinstance(result, dict)
    # Depending on implementation, may return error or same chart
    # We'll check it still returns a dict (no crash)
    assert 'chart' in result or 'error' in result

# ---------- BILLBOARD_200 ----------
def test_billboard_200_normal(music_instance):
    result = music_instance.BILLBOARD_200(date='2023-01-01', range='1-10')
    assert isinstance(result, dict)
    assert 'chart' in result
    assert len(result['chart']) == 1

def test_billboard_200_missing_param(music_instance):
    """Test with empty range (edge case)"""
    result = music_instance.BILLBOARD_200(date='2023-01-01', range='')
    assert isinstance(result, dict)

# ---------- Billboard_200_2 ----------
def test_billboard_200_2_normal(music_instance):
    result = music_instance.Billboard_200_2(date='2023-01-01')
    assert isinstance(result, dict)
    assert 'chart' in result
    assert len(result['chart']) == 1

def test_billboard_200_2_invalid_date(music_instance):
    """Test with None date (error case)"""
    result = music_instance.Billboard_200_2(date=None)
    assert isinstance(result, dict)
    assert 'error' in result or 'chart' in result  # gracefully handles

# ---------- Billboard_Global_Excl_US ----------
def test_billboard_global_excl_us_normal(music_instance):
    result = music_instance.Billboard_Global_Excl_US(date='2023-01-01')
    assert isinstance(result, dict)
    # Config has empty list for this
    assert result.get('chart') == []

def test_billboard_global_excl_us_missing_date(music_instance):
    """Test with missing date (None)"""
    result = music_instance.Billboard_Global_Excl_US(date=None)
    assert isinstance(result, dict)
    assert 'error' in result

# ---------- Billboard_Hot_100 ----------
def test_billboard_hot_100_normal(music_instance):
    result = music_instance.Billboard_Hot_100()
    assert isinstance(result, dict)
    assert 'chart' in result
    assert len(result['chart']) == 1

# ---------- Catalog_Albums ----------
def test_catalog_albums_normal(music_instance):
    result = music_instance.Catalog_Albums()
    assert isinstance(result, dict)
    assert 'chart' in result
    assert len(result['chart']) == 1

# ---------- Get_Channels ----------
def test_get_channels_normal(music_instance):
    result = music_instance.Get_Channels()
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(ch, dict) for ch in result)

def test_get_channels_empty(music_instance):
    """Test when config has empty channels list"""
    obj = MusicTools(initial_config={'videos': {}, 'charts': {}, 'albums': {}, 'search': {}, 'channels': []})
    result = obj.Get_Channels()
    assert isinstance(result, list)
    assert result == []

# ---------- Greatest_of_All_Time_Hot_100_Songs ----------
def test_greatest_of_all_time_hot_100_songs_normal(music_instance):
    result = music_instance.Greatest_of_All_Time_Hot_100_Songs()
    assert isinstance(result, dict)
    assert 'chart' in result
    assert len(result['chart']) == 1

# ---------- Independent_Albums ----------
def test_independent_albums_normal(music_instance):
    result = music_instance.Independent_Albums()
    assert isinstance(result, dict)
    assert 'chart' in result
    assert len(result['chart']) == 1

# ---------- List_User_Albums ----------
def test_list_user_albums_normal(music_instance):
    result = music_instance.List_User_Albums(user='spotify_user_1')
    assert isinstance(result, dict)
    assert 'albums' in result
    assert result['albums'] == ['album1', 'album2']

def test_list_user_albums_unknown_user(music_instance):
    result = music_instance.List_User_Albums(user='non_existent')
    assert isinstance(result, dict)
    assert 'error' in result

def test_list_user_albums_empty_user(music_instance):
    result = music_instance.List_User_Albums(user='empty_user')
    assert isinstance(result, dict)
    assert result['albums'] == []

# ---------- New_releases ----------
def test_new_releases_normal(music_instance):
    result = music_instance.New_releases(country='US')
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]['album'] == 'Album1'

def test_new_releases_missing_country(music_instance):
    result = music_instance.New_releases(country='GB')
    assert isinstance(result, list)
    assert result == []

def test_new_releases_invalid_country(music_instance):
    result = music_instance.New_releases(country='XX')
    assert isinstance(result, list)
    assert result == []  # or error? Config has only US/GB

# ---------- User_details ----------
def test_user_details_normal(music_instance):
    result = music_instance.User_details(user_id='spotify_user_1')
    assert isinstance(result, dict)
    assert result['display_name'] == 'John Doe'
    assert result['id'] == 'spotify_user_1'

def test_user_details_unknown_user(music_instance):
    result = music_instance.User_details(user_id='unknown_user')
    assert isinstance(result, dict)
    # unknown_user has empty dict; might return error
    assert 'error' in result or 'display_name' not in result

def test_user_details_nonexistent(music_instance):
    result = music_instance.User_details(user_id='does_not_exist')
    assert isinstance(result, dict)
    assert 'error' in result

# ---------- Year_End_Billboard_Global_200 ----------
def test_year_end_billboard_global_200_normal(music_instance):
    result = music_instance.Year_End_Billboard_Global_200(year=2022)
    assert isinstance(result, dict)
    assert 'chart' in result
    assert len(result['chart']) == 1

def test_year_end_billboard_global_200_invalid_year(music_instance):
    """Test with negative year (edge case)"""
    result = music_instance.Year_End_Billboard_Global_200(year=-1)
    assert isinstance(result, dict)
    # Should not crash; may return same data or error
    assert 'chart' in result or 'error' in result

# ---------- Year_End_Top_Artists ----------
def test_year_end_top_artists_normal(music_instance):
    result = music_instance.Year_End_Top_Artists(year=2022)
    assert isinstance(result, dict)
    assert 'chart' in result
    assert len(result['chart']) == 1

def test_year_end_top_artists_missing_year(music_instance):
    """Test with None year (error case)"""
    result = music_instance.Year_End_Top_Artists(year=None)
    assert isinstance(result, dict)
    assert 'error' in result

# ---------- auto_complete ----------
def test_auto_complete_normal(music_instance):
    result = music_instance.auto_complete(term='hello')
    assert isinstance(result, dict)
    assert 'suggestions' in result
    assert len(result['suggestions']) == 2

def test_auto_complete_empty_term(music_instance):
    """Test with empty term (edge case)"""
    result = music_instance.auto_complete(term='')
    assert isinstance(result, dict)
    assert 'suggestions' in result
    # Should return all? or empty? Config has 'empty' key; term '' might not match
    # We check it doesn't crash

def test_auto_complete_no_matches(music_instance):
    result = music_instance.auto_complete(term='nonexistent')
    assert isinstance(result, dict)
    # Return empty list or error
    suggestions = result.get('suggestions', [])
    assert isinstance(suggestions, list)

# ---------- boy_groups ----------
def test_boy_groups_normal(music_instance):
    result = music_instance.boy_groups(q='BTS')
    assert isinstance(result, dict)
    assert 'groups' in result
    assert len(result['groups']) >= 1  # 'BTS' is in list

def test_boy_groups_no_match(music_instance):
    result = music_instance.boy_groups(q='Nonexistent')
    assert isinstance(result, dict)
    assert 'groups' in result
    assert result['groups'] == []

# ---------- girl_groups ----------
def test_girl_groups_normal(music_instance):
    result = music_instance.girl_groups(q='TWICE')
    assert isinstance(result, dict)
    assert 'groups' in result
    assert 'TWICE' in result['groups']

def test_girl_groups_empty_q(music_instance):
    """Test with empty string (edge case)"""
    result = music_instance.girl_groups(q='')
    assert isinstance(result, dict)
    # Should return all groups or empty
    assert 'groups' in result

# ---------- random_boy_group ----------
def test_random_boy_group_normal(music_instance):
    result = music_instance.random_boy_group()
    assert isinstance(result, dict)
    assert 'group' in result
    assert result['group'] in ['BTS', 'EXO']

def test_random_boy_group_no_groups(music_instance):
    """Test when no boy groups configured"""
    obj = MusicTools(initial_config={'search': {'boy_groups': []}})
    result = obj.random_boy_group()
    assert isinstance(result, dict)
    assert 'error' in result

# ---------- random_song_song_s_album_information_out_of_artist ----------
def test_random_song_song_s_album_information_out_of_artist_normal(music_instance):
    result = music_instance.random_song_song_s_album_information_out_of_artist(artist='Taylor Swift')
    assert isinstance(result, dict)
    assert 'artist' in result
    assert 'album' in result
    assert 'song' in result

def test_random_song_song_s_album_information_out_of_artist_unknown(music_instance):
    result = music_instance.random_song_song_s_album_information_out_of_artist(artist='Unknown')
    assert isinstance(result, dict)
    assert 'error' in result

# ---------- random_song_from_a_specific_artist_and_specified_album ----------
def test_random_song_from_a_specific_artist_and_specified_album_normal(music_instance):
    result = music_instance.random_song_from_a_specific_artist_and_specified_album(artist='Taylor Swift', album='1989')
    assert isinstance(result, dict)
    assert 'song' in result
    assert result['song'] == 'Shake It Off'

def test_random_song_from_a_specific_artist_and_specified_album_mismatch(music_instance):
    """Test with nonexistent combination (edge case)"""
    result = music_instance.random_song_from_a_specific_artist_and_specified_album(artist='Taylor Swift', album='Red')
    assert isinstance(result, dict)
    # Should return error or fallback
    assert 'error' in result or 'song' in result

def test_random_song_from_a_specific_artist_and_specified_album_missing_params(music_instance):
    """Test with empty strings (edge case)"""
    result = music_instance.random_song_from_a_specific_artist_and_specified_album(artist='', album='')
    assert isinstance(result, dict)
    assert 'error' in result