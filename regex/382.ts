import { Regex } from '../models';

module.exports = {
    expression: new RegExp("^((?=.*we ))((?=.*obtain)|(?=.*collect))((?=.*sources)|(?=.*third parties)|(?=.*3rd parties)|(?=.*third-parties))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 382,
	name: "This service gathers information about you through third parties"
} as Regex;